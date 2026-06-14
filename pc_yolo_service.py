import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pc_yolo_config.json"
DEFAULT_CLASS_NAME_MAP = {
    "zangwu": "zang_wu",
    "zhezhou": "zhe_zhou",
    "zehzhou": "zhe_zhou",
    "lvpian": "zheng_chang",
}


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)
    model_path = Path(config.get("model_path", "train8/weights/best.pt"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    config["model_path"] = str(model_path)
    return config


def decode_image(file_storage):
    data = file_storage.read()
    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image file")
    return image


def to_size(width_px, height_px, config):
    size = {
        "width_px": float(width_px),
        "height_px": float(height_px),
        "width_mm": None,
        "height_mm": None,
        "display_text": "",
        "calibrated": False,
    }
    mm_x = config.get("mm_per_pixel_x")
    mm_y = config.get("mm_per_pixel_y")
    if mm_x is not None and mm_y is not None:
        size["width_mm"] = float(width_px) * float(mm_x)
        size["height_mm"] = float(height_px) * float(mm_y)
        size["display_text"] = "{:.2f}x{:.2f}mm".format(size["width_mm"], size["height_mm"])
        size["calibrated"] = True
    else:
        size["display_text"] = "{:.2f}x{:.2f}px(uncalibrated)".format(size["width_px"], size["height_px"])
    return size


def bbox_to_payload(bbox, config):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    size = to_size(max(0.0, x2 - x1), max(0.0, y2 - y1), config)
    return {
        "loc": [x1, y1, x2, y2],
        "size": size,
        "width_px": size["width_px"],
        "height_px": size["height_px"],
        "width_mm": size["width_mm"],
        "height_mm": size["height_mm"],
        "display_text": size["display_text"],
        "calibrated": size["calibrated"],
    }


def contour_bbox(mask, min_area, max_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if area > best_area:
            best_area = area
            best = contour
    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(best)
    return [float(x), float(y), float(x + w), float(y + h)]


def detect_plate_bbox(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * float(config.get("plate_min_area_ratio", 0.02))
    max_area = image_area * float(config.get("plate_max_area_ratio", 0.98))

    candidates = []
    for mask in (binary, cv2.bitwise_not(binary)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bbox = contour_bbox(mask, min_area, max_area)
        if bbox:
            x1, y1, x2, y2 = bbox
            candidates.append((max(0.0, (x2 - x1) * (y2 - y1)), bbox))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    h, w = image.shape[:2]
    return [0.0, 0.0, float(w), float(h)]


def class_name(model, class_id):
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(int(class_id), int(class_id)))
    return str(names[int(class_id)])


def mapped_class_name(raw_name, config):
    mapping = dict(DEFAULT_CLASS_NAME_MAP)
    mapping.update(config.get("class_name_map", {}))
    return mapping.get(raw_name, raw_name)


def mimo_settings(config):
    mimo = config.get("mimo", {}) or {}
    api_key = os.environ.get("MIMO_API_KEY") or mimo.get("api_key", "")
    return {
        "api_key": str(api_key).strip(),
        "base_url": str(mimo.get("base_url", "https://api.xiaomimimo.com/v1/chat/completions")).strip(),
        "model": str(mimo.get("model", "mimo-v2.5")).strip(),
        "timeout": float(mimo.get("timeout", 30)),
        "max_completion_tokens": int(mimo.get("max_completion_tokens", 1024)),
    }


def image_bytes_to_data_url(image_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")


def mimo_prompt(measurement):
    measurement_text = json.dumps(measurement or {}, ensure_ascii=False)
    if len(measurement_text) > 4000:
        measurement_text = measurement_text[:4000] + "..."
    return (
        "你是工业铝片缺陷质检智能体。请结合图片和尺寸数据分析当前铝片缺陷。"
        "请用中文输出，内容包括：1. 铝片整体情况；2. 缺陷类型；3. 缺陷尺寸和面积解读；"
        "4. 是否建议判为NG；5. 给现场操作员的处理建议。"
        "不要编造没有看到的数据；如果尺寸数据缺失，请明确说明。\n\n"
        "当前尺寸数据如下：\n{}".format(measurement_text)
    )


def create_app(config):
    app = Flask(__name__)
    CORS(app, supports_credentials=True)
    model = YOLO(config["model_path"])
    aluminum_names = set(config.get("aluminum_class_names", []))
    normal_names = set(config.get("normal_class_names", ["zheng_chang"]))

    @app.route("/", methods=["GET"])
    def index():
        return jsonify(
            {
                "status": "ok",
                "service": "pc-yolo-service",
                "message": "Open /health to check status. Raspberry Pi should POST images to /predict.",
                "predict_url": "/predict",
                "health_url": "/health",
            }
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_path": config["model_path"],
                "names": model.names,
                "class_name_map": config.get("class_name_map", {}),
            }
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        start = time.time()
        file_storage = request.files.get("image_file") or request.files.get("file") or request.files.get("image")
        if file_storage is None:
            return jsonify({"error": "missing image_file"}), 400

        image = decode_image(file_storage)
        result = model.predict(
            image,
            conf=float(config.get("confidence", 0.25)),
            iou=float(config.get("iou", 0.7)),
            imgsz=int(config.get("image_size", 640)),
            verbose=False,
        )[0]

        detections = []
        aluminum_candidates = []
        elapsed_ms = (time.time() - start) * 1000.0

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_name = class_name(model, cls_id)
                name = mapped_class_name(raw_name, config)
                conf = float(box.conf[0])
                bbox = [float(v) for v in box.xyxy[0].tolist()]
                payload = bbox_to_payload(bbox, config)

                is_aluminum = raw_name in aluminum_names or name in aluminum_names
                if is_aluminum:
                    aluminum_candidates.append((conf, payload))

                if is_aluminum and name not in normal_names:
                    continue

                payload.update(
                    {
                        "class_name": name,
                        "raw_class_name": raw_name,
                        "score": conf,
                        "prediction_time": elapsed_ms,
                    }
                )
                detections.append(payload)

        if aluminum_candidates:
            aluminum_candidates.sort(key=lambda item: item[0], reverse=True)
            aluminum = aluminum_candidates[0][1]
        else:
            aluminum = bbox_to_payload(detect_plate_bbox(image, config), config)

        return jsonify(
            {
                "len": len(detections),
                "result": detections,
                "aluminum": aluminum,
                "prediction_time": elapsed_ms,
            }
        )

    @app.route("/mimo_analyze_defect", methods=["POST"])
    def mimo_analyze_defect():
        settings = mimo_settings(config)
        if not settings["api_key"]:
            return jsonify(
                {
                    "status": "missing_key",
                    "message": "PC MiMo API Key 未配置。请在 pc_yolo_config.json 的 mimo.api_key 中填写 key，或设置 MIMO_API_KEY 环境变量。",
                }
            ), 400

        file_storage = request.files.get("image_file") or request.files.get("file") or request.files.get("image")
        if file_storage is None:
            return jsonify({"status": "no_image", "message": "没有收到检测图片。"}), 400

        image_bytes = file_storage.read()
        measurement_text = request.form.get("measurement", "{}")
        try:
            measurement = json.loads(measurement_text)
        except Exception:
            measurement = {}

        payload = {
            "model": settings["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "你是专业的工业视觉质检助手，当前日期是2026年06月04日。",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_bytes_to_data_url(image_bytes)},
                        },
                        {
                            "type": "text",
                            "text": mimo_prompt(measurement),
                        },
                    ],
                },
            ],
            "max_completion_tokens": settings["max_completion_tokens"],
        }

        try:
            response = requests.post(
                settings["base_url"],
                headers={"api-key": settings["api_key"], "Content-Type": "application/json"},
                json=payload,
                timeout=settings["timeout"],
            )
            response.raise_for_status()
            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {})
            return jsonify(
                {
                    "status": "ok",
                    "model": settings["model"],
                    "analysis": message.get("content", ""),
                    "measurement_summary": measurement.get("summary", ""),
                    "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        except Exception as exc:
            return jsonify({"status": "error", "message": "MiMo 分析失败：{}".format(str(exc))}), 500

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config)
    app.run(host=config.get("host", "0.0.0.0"), port=int(config.get("port", 8080)), debug=False, threaded=True)


if __name__ == "__main__":
    main()
