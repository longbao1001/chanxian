import configparser
import datetime
import math
import uuid

import cv2
import numpy as np


DEFAULT_STATE = {
    "capture_id": "",
    "timestamp": "",
    "calibration": {
        "ready": False,
        "mode": "fixed_camera_reference",
        "pixels_per_mm": 0.0,
        "confidence": 0.0,
        "message": "not calibrated",
    },
    "sheet": {
        "diameter_mm": 0.0,
        "length_mm": 0.0,
        "width_mm": 0.0,
        "angle_deg": 0.0,
        "confidence": 0.0,
    },
    "defects": [],
    "grasp": {
        "mode": "fallback",
        "target_x_mm": 0.0,
        "target_y_mm": 0.0,
        "target_angle_deg": 0.0,
        "fallback_slot": "OK",
    },
}


def _now_text():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _round1(value):
    try:
        return round(float(value), 1)
    except Exception:
        return 0.0


def _round2(value):
    try:
        return round(float(value), 2)
    except Exception:
        return 0.0


def _ensure_section(cf):
    if not cf.has_section("Measurement"):
        cf.add_section("Measurement")
    defaults = {
        "calibration_mode": "fixed_camera_reference",
        "calibration_diameter_mm": "100",
        "pixels_per_mm": "0",
        "measurement_roi": "",
        "calibrated": "0",
    }
    changed = False
    for key, value in defaults.items():
        if not cf.has_option("Measurement", key):
            cf.set("Measurement", key, value)
            changed = True
    if cf.get("Measurement", "calibration_mode", fallback="") == "circle_100mm":
        cf.set("Measurement", "calibration_mode", "fixed_camera_reference")
        changed = True
    return changed


def ensure_measurement_config(config_path, cf):
    changed = _ensure_section(cf)
    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            cf.write(f)


def _cfg_float(cf, key, default=0.0):
    try:
        return cf.getfloat("Measurement", key)
    except Exception:
        return default


def _cfg_bool(cf, key, default=False):
    try:
        return cf.getboolean("Measurement", key)
    except Exception:
        return default


def _sheet_radius_range_from_config(cf):
    ppm = _cfg_float(cf, "pixels_per_mm", 0.0)
    if ppm <= 0:
        return 0.0, 0.0, 0.0
    try:
        min_mm = cf.getfloat("Capture", "min_sheet_diameter_mm", fallback=80.0)
        max_mm = cf.getfloat("Capture", "max_sheet_diameter_mm", fallback=120.0)
    except Exception:
        min_mm, max_mm = 80.0, 120.0
    if min_mm <= 0 or max_mm <= 0:
        return 0.0, 0.0
    if min_mm > max_mm:
        min_mm, max_mm = max_mm, min_mm
    min_radius = ppm * min_mm / 2.0
    max_radius = ppm * max_mm / 2.0
    expected_radius = (min_radius + max_radius) / 2.0
    try:
        ref_mm = cf.getfloat("Measurement", "calibration_diameter_mm", fallback=0.0)
        if ref_mm > 0:
            expected_radius = ppm * ref_mm / 2.0
    except Exception:
        pass
    return min_radius, max_radius, expected_radius


def _parse_roi(cf, shape):
    try:
        text = cf.get("Measurement", "measurement_roi").strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        x, y, w, h = [int(float(v.strip())) for v in text.split(",")[:4]]
    except Exception:
        return None
    image_h, image_w = shape[:2]
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    return x, y, w, h


def _to_gray(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _crop_roi(image, cf):
    roi = _parse_roi(cf, image.shape)
    if not roi:
        return image, (0, 0)
    x, y, w, h = roi
    return image[y:y + h, x:x + w], (x, y)


def _candidate_contour(gray, radius_range=None):
    if gray is None or gray.size == 0:
        return None, 0.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    masks = []
    _, m1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.extend([m1, m2])
    # Strong highlights on brushed aluminum can make Otsu select the bright
    # inner reflection instead of the actual disk edge. Add a small set of
    # absolute and percentile thresholds so the outer rim is considered too.
    extra_thresholds = set()
    try:
        for val in (45, 55, 65, 75, 85, 95, 110, 125):
            extra_thresholds.add(int(val))
        for pct in (58, 64, 70, 76, 82):
            extra_thresholds.add(int(np.percentile(blur, pct)))
    except Exception:
        extra_thresholds = set()
    for thr in sorted(v for v in extra_thresholds if 4 <= v <= 245):
        _, mt = cv2.threshold(blur, int(thr), 255, cv2.THRESH_BINARY)
        masks.append(mt)
    kernel = np.ones((3, 3), np.uint8)
    best = None
    best_score = 0.0
    image_area = float(gray.shape[0] * gray.shape[1])
    min_radius, max_radius, expected_radius = radius_range or (0.0, 0.0, 0.0)
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.02:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = min(1.0, (4.0 * math.pi * area) / (perimeter * perimeter))
            (_, _), radius = cv2.minEnclosingCircle(contour)
            # Reject frame-sized contours. During calibration a bright sheet on
            # a dark belt can make thresholding select the whole image border,
            # which previously wrote a bogus 9+ px/mm scale.
            if radius > min(gray.shape[:2]) * 0.47:
                continue
            if area > image_area * 0.62:
                continue
            if min_radius > 0 and max_radius > 0:
                if radius < min_radius * 0.92 or radius > max_radius * 1.08:
                    continue
            area_ratio = min(1.0, area / max(image_area * 0.45, 1.0))
            radius_score = 1.0
            if expected_radius > 0:
                radius_err = abs(float(radius) - expected_radius) / max(expected_radius, 1.0)
                radius_score = max(0.18, 1.0 - radius_err * 2.2)
            # Prefer the real outer rim over a smaller bright blob: the area
            # term is deliberately strong, while the calibrated radius remains
            # only a soft prior so 80-120mm sheets are still measurable.
            score = (0.54 * circularity + 0.34 * area_ratio + 0.12 * radius_score) * radius_score
            if score > best_score:
                best_score = score
                best = contour
    return best, best_score


def _sheet_metrics(image, cf, pixels_per_mm):
    gray = _to_gray(image)
    if gray is None:
        return DEFAULT_STATE["sheet"].copy()
    work, offset = _crop_roi(gray, cf)
    radius_range = _sheet_radius_range_from_config(cf)
    contour, confidence = _candidate_contour(work, radius_range)
    if contour is None or pixels_per_mm <= 0:
        return DEFAULT_STATE["sheet"].copy()
    offset_arr = np.array(offset, dtype=np.int32)
    contour = contour + offset_arr
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    diameter_px = radius * 2.0
    if rw > 1 and rh > 1:
        diameter_px = max(diameter_px, (rw + rh) / 2.0)
    diameter_mm = diameter_px / pixels_per_mm
    return {
        "diameter_mm": _round1(diameter_mm),
        "length_mm": _round1(diameter_mm),
        "width_mm": _round1(diameter_mm),
        "angle_deg": _round1(rect[2]),
        "confidence": _round2(confidence),
        "radius_px": _round2(radius),
        "center_px": [_round2(cx), _round2(cy)],
        "measurement_source": "detected_circle",
    }


def _bbox_from_detection(item):
    loc = item.get("loc") or item.get("bbox") or []
    if len(loc) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in loc[:4]]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2



def _score_rule_dimensions(bbox, pixels_per_mm):
    x1, y1, x2, y2 = bbox
    width_px = max(0, x2 - x1)
    height_px = max(0, y2 - y1)
    if width_px <= 0 or height_px <= 0 or pixels_per_mm <= 0:
        return None
    width_mm_raw = width_px / pixels_per_mm
    height_mm_raw = height_px / pixels_per_mm
    width_mm = _round1(width_mm_raw)
    height_mm = _round1(height_mm_raw)
    return {
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "length_mm": _round1(max(width_mm_raw, height_mm_raw)),
        "display_width_mm": _round1(min(width_mm_raw, height_mm_raw)),
        "area_mm2": _round2(width_mm_raw * height_mm_raw),
    }


def _merge_bboxes(items):
    boxes = [item["bbox"] for item in items if item.get("bbox")]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _defect_metrics(image, detections, pixels_per_mm):
    if pixels_per_mm <= 0:
        return []
    grouped = {}
    singles = []
    for item in detections or []:
        class_name = item.get("class_name", "")
        if class_name == "zheng_chang":
            continue
        bbox = _bbox_from_detection(item)
        if not bbox:
            continue
        refine_mode = item.get("refine_mode") or item.get("box_mode") or "single"
        row = {
            "type": class_name,
            "score": float(item.get("score", 0.0) or 0.0),
            "bbox": bbox,
            "refined": bool(item.get("refined")),
            "refine_mode": refine_mode,
            "quality_flag": item.get("quality_flag", "ok"),
            "decision_source": item.get("decision_source", refine_mode),
            "trigger_reason": item.get("trigger_reason", ""),
        }
        for key in ("measurement_rect_mode", "rotated_rect_px", "component_id", "box_width_mm", "box_height_mm", "width_mm", "height_mm", "length_mm", "area_mm2", "area_method", "debug", "ssd_confidence", "yolo_confidence", "final_confidence", "detector_role"):
            if key in item:
                row[key] = item.get(key)
        if class_name in ("zang_wu", "zhe_zhou") and not row["refined"]:
            grouped.setdefault(class_name, []).append(row)
        else:
            singles.append(row)

    defects = []
    for class_name, rows in grouped.items():
        bbox = _merge_bboxes(rows)
        dims = _score_rule_dimensions(bbox, pixels_per_mm) if bbox else None
        if not dims:
            continue
        x1, y1, x2, y2 = bbox
        score = max(row["score"] for row in rows) if rows else 0.0
        defects.append({
            "type": class_name,
            "score": _round2(score),
            "length_mm": dims["length_mm"],
            "width_mm": dims["display_width_mm"],
            "height_mm": dims["height_mm"],
            "box_width_mm": dims["width_mm"],
            "box_height_mm": dims["height_mm"],
            "area_mm2": dims["area_mm2"],
            "area_method": "min_rect_score_rule",
            "box_mode": "merged_same_type" if len(rows) > 1 else "single",
            "quality_flag": "review_classification" if any(row.get("quality_flag") != "ok" for row in rows) else "ok",
            "decision_source": rows[0].get("decision_source", "model_fallback") if rows else "model_fallback",
            "trigger_reason": rows[0].get("trigger_reason", "") if rows else "",
            "bbox_px": [x1, y1, x2, y2],
            "debug": {
                "pixels_per_mm": _round2(pixels_per_mm),
                "bbox_px_width": dims["width_px"],
                "bbox_px_height": dims["height_px"],
                "source_count": len(rows),
            },
        })

    for row in singles:
        bbox = row["bbox"]
        dims = _score_rule_dimensions(bbox, pixels_per_mm)
        if not dims:
            continue
        x1, y1, x2, y2 = bbox
        has_measurement_rect = bool(row.get("measurement_rect_mode") and row.get("box_width_mm") and row.get("box_height_mm") and row.get("area_mm2"))
        defect = {
            "type": row["type"],
            "score": _round2(row["score"]),
            "length_mm": row.get("length_mm", dims["length_mm"]) if has_measurement_rect else dims["length_mm"],
            "width_mm": row.get("width_mm", dims["display_width_mm"]) if has_measurement_rect else dims["display_width_mm"],
            "height_mm": row.get("height_mm", dims["height_mm"]) if has_measurement_rect else dims["height_mm"],
            "box_width_mm": row.get("box_width_mm", dims["width_mm"]) if has_measurement_rect else dims["width_mm"],
            "box_height_mm": row.get("box_height_mm", dims["height_mm"]) if has_measurement_rect else dims["height_mm"],
            "area_mm2": row.get("area_mm2", dims["area_mm2"]) if has_measurement_rect else dims["area_mm2"],
            "area_method": (row.get("area_method") or "min_area_rect_score_rule") if has_measurement_rect else ("min_rect_score_rule" if row.get("type") in ("zang_wu", "zhe_zhou") else "bbox_fallback"),
            "box_mode": row.get("refine_mode") or "single",
            "quality_flag": row.get("quality_flag", "ok"),
            "decision_source": row.get("decision_source", row.get("refine_mode") or "model_fallback"),
            "trigger_reason": row.get("trigger_reason", ""),
            "bbox_px": [x1, y1, x2, y2],
            "debug": {
                "pixels_per_mm": _round2(pixels_per_mm),
                "bbox_px_width": dims["width_px"],
                "bbox_px_height": dims["height_px"],
                "source_count": 1,
            },
        }
        if has_measurement_rect:
            defect["measurement_rect_mode"] = row.get("measurement_rect_mode")
            if row.get("rotated_rect_px"):
                defect["rotated_rect_px"] = row.get("rotated_rect_px")
            if row.get("component_id") is not None:
                defect["component_id"] = row.get("component_id")
            if isinstance(row.get("debug"), dict):
                defect["debug"].update(row.get("debug"))
        for key in ("ssd_confidence", "yolo_confidence", "final_confidence", "detector_role"):
            if row.get(key) is not None:
                defect[key] = row.get(key)
        defects.append(defect)
    return defects

def build_measurement_state(image, detections, cf, capture_id=None, timestamp=None,
                            fallback_slot="OK", predict_fps=0.0):
    _ensure_section(cf)
    capture_id = capture_id or str(uuid.uuid4())
    timestamp = timestamp or _now_text()
    pixels_per_mm = _cfg_float(cf, "pixels_per_mm", 0.0)
    calibrated = _cfg_bool(cf, "calibrated", False) and pixels_per_mm > 0
    state = {
        "capture_id": capture_id,
        "timestamp": timestamp,
        "calibration": {
            "ready": bool(calibrated),
            "mode": "fixed_camera_reference",
            "pixels_per_mm": _round2(pixels_per_mm),
            "confidence": 1.0 if calibrated else 0.0,
            "message": "ok" if calibrated else "need fixed-camera calibration",
        },
        "sheet": DEFAULT_STATE["sheet"].copy(),
        "defects": [],
        "grasp": {
            "mode": "fallback",
            "target_x_mm": 0.0,
            "target_y_mm": 0.0,
            "target_angle_deg": 0.0,
            "fallback_slot": fallback_slot,
        },
        "predict_fps": _round1(predict_fps),
        "measurement_version": "score_rect_v2",
    }
    if calibrated:
        state["sheet"] = _sheet_metrics(image, cf, pixels_per_mm)
        state["defects"] = _defect_metrics(image, detections, pixels_per_mm)
        if state["sheet"]["confidence"] > 0:
            state["calibration"]["confidence"] = state["sheet"]["confidence"]
    return state


def calibrate_from_image(image, cf, config_path, reference_diameter_mm=None):
    _ensure_section(cf)
    gray = _to_gray(image)
    if gray is None:
        return {"ok": False, "message": "no image"}
    work, _ = _crop_roi(gray, cf)
    contour, confidence = _candidate_contour(work)
    if contour is None:
        return {"ok": False, "message": "circle contour not found", "confidence": 0.0}
    (_, _), radius = cv2.minEnclosingCircle(contour)
    diameter_px = radius * 2.0
    try:
        diameter_mm = float(reference_diameter_mm) if reference_diameter_mm is not None else _cfg_float(cf, "calibration_diameter_mm", 100.0)
    except Exception:
        diameter_mm = _cfg_float(cf, "calibration_diameter_mm", 100.0)
    if diameter_px <= 0 or diameter_mm <= 0:
        return {"ok": False, "message": "invalid circle size", "confidence": _round2(confidence)}
    pixels_per_mm = diameter_px / diameter_mm
    cf.set("Measurement", "calibration_mode", "fixed_camera_reference")
    cf.set("Measurement", "calibration_diameter_mm", "{:.3f}".format(diameter_mm).rstrip("0").rstrip("."))
    cf.set("Measurement", "pixels_per_mm", "{:.6f}".format(pixels_per_mm))
    cf.set("Measurement", "calibrated", "1")
    with open(config_path, "w", encoding="utf-8") as f:
        cf.write(f)
    return {
        "ok": True,
        "message": "calibrated",
        "mode": "fixed_camera_reference",
        "diameter_px": _round2(diameter_px),
        "diameter_mm": _round1(diameter_mm),
        "pixels_per_mm": _round2(pixels_per_mm),
        "confidence": _round2(confidence),
    }
