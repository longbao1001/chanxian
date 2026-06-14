import math

import cv2
import numpy as np


def _round_score(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _bbox(item):
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


def _clip_box(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _gray(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _pixels_per_mm_from_config(cf):
    if cf is None:
        return 0.0
    try:
        return float(cf.get("Measurement", "pixels_per_mm") or 0)
    except Exception:
        return 0.0


def _sheet_radius_range_from_config(cf):
    ppm = _pixels_per_mm_from_config(cf)
    if ppm <= 0:
        return 0.0, 0.0, 0.0
    try:
        min_mm = float(cf.get("Capture", "min_sheet_diameter_mm", fallback="80") or 80)
        max_mm = float(cf.get("Capture", "max_sheet_diameter_mm", fallback="120") or 120)
    except Exception:
        min_mm, max_mm = 80.0, 120.0
    if min_mm <= 0 or max_mm <= 0:
        return 0.0, 0.0, 0.0
    if min_mm > max_mm:
        min_mm, max_mm = max_mm, min_mm
    min_radius = ppm * min_mm / 2.0
    max_radius = ppm * max_mm / 2.0
    expected_radius = (min_radius + max_radius) / 2.0
    try:
        cal_mm = float(cf.get("Measurement", "calibration_diameter_mm", fallback="0") or 0)
        if cal_mm > 0:
            expected_radius = ppm * cal_mm / 2.0
    except Exception:
        pass
    return min_radius, max_radius, expected_radius


def _center_roi_from_config(cf, shape):
    h, w = shape[:2]
    default = (230, 150, 590, 380)
    try:
        raw = cf.get("Capture", "center_sheet_roi", fallback="") if cf is not None and cf.has_section("Capture") else ""
        vals = [int(round(float(v.strip()))) for v in str(raw or "").split(",") if v.strip()]
        if len(vals) == 4:
            return tuple(vals)
    except Exception:
        pass
    x1, y1, x2, y2 = default
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _center_distance_to_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    tx = (x1 + x2) / 2.0
    ty = (y1 + y2) / 2.0
    return math.hypot(float(cx) - tx, float(cy) - ty)


def _max_center_distance_from_config(cf, radius):
    try:
        if cf is not None and cf.has_section("Capture"):
            return float(cf.get("Capture", "max_center_distance_px", fallback=str(max(135.0, radius * 0.78))) or max(135.0, radius * 0.78))
    except Exception:
        pass
    return max(135.0, radius * 0.78)


def _hough_sheet(gray, min_radius, max_radius, expected_radius):
    if gray is None or gray.size == 0 or min_radius <= 0 or max_radius <= 0:
        return None
    blur = cv2.medianBlur(gray, 5)
    min_r = max(25, int(round(min_radius * 0.92)))
    max_r = max(min_r + 4, int(round(max_radius * 1.08)))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(round(expected_radius * 0.9))),
        param1=80,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    h, w = gray.shape[:2]
    best = None
    best_score = -1.0
    for cx, cy, radius in np.round(circles[0, :]).astype(int):
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
        full_area = max(1.0, math.pi * float(radius) * float(radius))
        visible_ratio = np.count_nonzero(mask) / full_area
        if visible_ratio < 0.55:
            continue
        vals = gray[mask > 0]
        if vals.size < 50:
            continue
        radius_err = abs(float(radius) - expected_radius) / max(expected_radius, 1.0) if expected_radius > 0 else 0.0
        # The aluminum disk is brighter and more textured than the black belt.
        score = (float(np.mean(vals)) + 0.45 * float(np.std(vals))) * max(0.25, 1.0 - radius_err) * min(1.0, visible_ratio / 0.82)
        if score > best_score:
            contour = cv2.ellipse2Poly((int(cx), int(cy)), (int(radius), int(radius)), 0, 0, 360, 8).reshape((-1, 1, 2))
            best_score = score
            best = (float(cx), float(cy), float(radius), contour)
    return best


def _hough_sheet_candidates(gray, min_radius, max_radius, expected_radius):
    if gray is None or gray.size == 0 or min_radius <= 0 or max_radius <= 0:
        return []
    blur = cv2.medianBlur(gray, 5)
    min_r = max(25, int(round(min_radius * 0.92)))
    max_r = max(min_r + 4, int(round(max_radius * 1.08)))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(round(expected_radius * 0.9))),
        param1=80,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    candidates = []
    for cx, cy, radius in np.round(circles[0, :]).astype(int):
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
        vals = gray[mask > 0]
        if vals.size < 50:
            continue
        contour = cv2.ellipse2Poly((int(cx), int(cy)), (int(radius), int(radius)), 0, 0, 360, 8).reshape((-1, 1, 2))
        candidates.append((float(cx), float(cy), float(radius), contour))
    return candidates


def _find_sheet(gray, cf=None):
    if gray is None or gray.size == 0:
        return None
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    masks = []
    _, m1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.extend([m1, m2])
    best = None
    best_score = -1.0
    image_area = float(gray.shape[0] * gray.shape[1])
    kernel = np.ones((5, 5), np.uint8)
    min_radius, max_radius, expected_radius = _sheet_radius_range_from_config(cf)
    center_roi = _center_roi_from_config(cf, gray.shape)
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.035 or area > image_area * 0.45:
                continue
            peri = cv2.arcLength(contour, True)
            if peri <= 0:
                continue
            circularity = (4.0 * math.pi * area) / (peri * peri)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < 35:
                continue
            if min_radius > 0 and max_radius > 0:
                if radius < min_radius * 0.92 or radius > max_radius * 1.08:
                    continue
            center_dist = _center_distance_to_roi(cx, cy, center_roi)
            max_center_dist = _max_center_distance_from_config(cf, radius)
            if center_dist > max_center_dist * 1.35:
                continue
            score = circularity * min(area, image_area * 0.2)
            if expected_radius > 0:
                radius_err = abs(radius - expected_radius) / max(expected_radius, 1.0)
                score *= max(0.25, 1.0 - radius_err)
            score *= max(0.18, 1.0 - center_dist / max(max_center_dist * 1.45, 1.0))
            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(radius), contour)
    if min_radius > 0 and max_radius > 0:
        hough = _hough_sheet(gray, min_radius, max_radius, expected_radius)
        if hough is not None:
            hcx, hcy, hradius, _ = hough
            hdist = _center_distance_to_roi(hcx, hcy, center_roi)
            hlimit = _max_center_distance_from_config(cf, hradius)
            if hdist > hlimit * 1.25:
                hough = None
        if hough is not None:
            if best is None:
                best = hough
            else:
                _, _, contour_radius, _ = best
                _, _, hough_radius, _ = hough
                if abs(hough_radius - expected_radius) < abs(contour_radius - expected_radius) * 0.75:
                    best = hough
    return best


def _circle_mask(shape, circle, shrink=0.90):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not circle:
        return mask
    cx, cy, radius, _ = circle
    cv2.circle(mask, (int(round(cx)), int(round(cy))), max(1, int(round(radius * shrink))), 255, -1)
    return mask


def _circle_bounds(circle, shape, margin=0.04):
    if not circle:
        return None
    h, w = shape[:2]
    cx, cy, radius, _ = circle
    pad = max(2, int(round(radius * margin)))
    x1 = int(round(cx - radius - pad))
    y1 = int(round(cy - radius - pad))
    x2 = int(round(cx + radius + pad))
    y2 = int(round(cy + radius + pad))
    return _clip_box((x1, y1, x2, y2), shape)


def _intersect_box(a, b):
    if not a or not b:
        return None
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_mm(box, pixels_per_mm):
    if not box or pixels_per_mm <= 0:
        return 0.0, 0.0, 0.0
    x1, y1, x2, y2 = box
    w_mm = abs(x2 - x1) / pixels_per_mm
    h_mm = abs(y2 - y1) / pixels_per_mm
    return w_mm, h_mm, w_mm * h_mm


def _rotated_rect_from_mask(mask, offset=(0, 0), min_points=8):
    if mask is None:
        return None
    pts = cv2.findNonZero(mask.astype("uint8"))
    if pts is None or len(pts) < min_points:
        return None
    pts = pts.reshape(-1, 2).astype(np.float32)
    pts[:, 0] += float(offset[0])
    pts[:, 1] += float(offset[1])
    return cv2.minAreaRect(pts)


def _rotated_rect_payload(rect, shape, pixels_per_mm):
    if rect is None:
        return None
    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return None
    points = cv2.boxPoints(rect)
    clipped_points = []
    height, width = shape[:2]
    for x, y in points:
        clipped_points.append([
            round(float(max(0.0, min(width - 1.0, x))), 2),
            round(float(max(0.0, min(height - 1.0, y))), 2),
        ])
    xs = [p[0] for p in clipped_points]
    ys = [p[1] for p in clipped_points]
    bbox = _clip_box((int(math.floor(min(xs))), int(math.floor(min(ys))), int(math.ceil(max(xs))), int(math.ceil(max(ys)))), shape)
    long_px = float(max(w, h))
    short_px = float(min(w, h))
    long_mm = long_px / pixels_per_mm if pixels_per_mm > 0 else 0.0
    short_mm = short_px / pixels_per_mm if pixels_per_mm > 0 else 0.0
    return {
        "bbox": bbox,
        "points": clipped_points,
        "center": [round(float(cx), 2), round(float(cy), 2)],
        "angle": round(float(angle), 2),
        "width_px": round(short_px, 2),
        "height_px": round(long_px, 2),
        "width_mm": round(short_mm, 1),
        "height_mm": round(long_mm, 1),
        "area_mm2": round(short_mm * long_mm, 2),
    }


def _rect_payload_from_box(box, shape, pixels_per_mm):
    box = _clip_box(box, shape) if box else None
    if not box:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    width_px = float(max(1, x2 - x1))
    height_px = float(max(1, y2 - y1))
    short_px = min(width_px, height_px)
    long_px = max(width_px, height_px)
    short_mm = short_px / pixels_per_mm if pixels_per_mm > 0 else 0.0
    long_mm = long_px / pixels_per_mm if pixels_per_mm > 0 else 0.0
    return {
        "bbox": box,
        "points": [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]],
        "center": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
        "angle": 0.0,
        "width_px": round(short_px, 2),
        "height_px": round(long_px, 2),
        "width_mm": round(short_mm, 1),
        "height_mm": round(long_mm, 1),
        "area_mm2": round(short_mm * long_mm, 2),
    }


def _standard_wrinkle_payload(cross, circle, shape, pixels_per_mm):
    if not cross or not circle:
        return None
    cx, cy, radius, _ = circle
    try:
        ix, iy = float(cross[0]), float(cross[1])
    except Exception:
        return None
    # The contest reference frames the crossed wrinkle as one scoring region,
    # close to a square that covers the two fold bands and their influence
    # area. A tight line minAreaRect is visually too small for scoring.
    side = max(radius * 1.76, 72.0)
    side = min(side, radius * 1.90)
    # Keep the scoring square near the physical sheet center when glare shifts
    # the detected line intersection a little to one side.
    max_shift = radius * 0.18
    dx = max(-max_shift, min(max_shift, ix - cx))
    dy = max(-max_shift, min(max_shift, iy - cy))
    sx = cx + dx * 0.72
    sy = cy + dy * 0.72
    half = side / 2.0
    box = (
        int(round(sx - half)),
        int(round(sy - half)),
        int(round(sx + half)),
        int(round(sy + half)),
    )
    payload = _rect_payload_from_box(box, shape, pixels_per_mm)
    if payload:
        payload.setdefault("debug", {})
        payload["debug"].update({
            "standard_wrinkle_square": True,
            "cross_point": [round(ix, 2), round(iy, 2)],
            "side_px": round(side, 2),
        })
    return payload


def _competition_wrinkle_payload(cross, circle, shape, pixels_per_mm, base_payload=None, source="local_cross", seed_box=None):
    payload = _standard_wrinkle_payload(cross, circle, shape, pixels_per_mm)
    if not payload:
        return base_payload
    if seed_box:
        try:
            cx, cy, radius, _ = circle
            pad = max(8, int(round(radius * 0.035)))
            sx1, sy1, sx2, sy2 = [int(round(float(v))) for v in seed_box[:4]]
            seed = (sx1 - pad, sy1 - pad, sx2 + pad, sy2 + pad)
            merged = _merge_boxes(payload.get("bbox"), seed)
            bounds = _circle_bounds(circle, shape, 0.01)
            merged = _intersect_box(merged, bounds) or _clip_box(merged, shape) or payload.get("bbox")
            if merged and _box_center_distance_ratio(merged, circle) <= 0.68 and _mask_overlap_ratio(merged, circle, shape, 0.98) >= 0.42:
                payload = _rect_payload_from_box(merged, shape, pixels_per_mm) or payload
        except Exception:
            pass
    payload["measurement_rect_mode"] = "wrinkle_competition_cross_rect"
    payload.setdefault("debug", {})
    payload["debug"].update({
        "wrinkle_rect_rule": "competition_cross_scoring_rect",
        "competition_source": source,
        "competition_seed_union": bool(seed_box),
    })
    if base_payload and isinstance(base_payload, dict):
        payload["debug"]["base_bbox"] = base_payload.get("bbox")
        payload["debug"]["base_width_mm"] = base_payload.get("width_mm")
        payload["debug"]["base_height_mm"] = base_payload.get("height_mm")
        payload["debug"]["base_rule"] = (base_payload.get("debug") or {}).get("wrinkle_rect_rule")
    return payload


def _apply_measurement_rect(item, payload, mode="min_area_rect"):
    if not item or not payload or not payload.get("bbox"):
        return item
    item["loc"] = [int(v) for v in payload["bbox"]]
    item["rotated_rect_px"] = payload["points"]
    item["measurement_rect_mode"] = payload.get("measurement_rect_mode") or mode
    item["box_width_mm"] = payload["width_mm"]
    item["box_height_mm"] = payload["height_mm"]
    item["width_mm"] = payload["width_mm"]
    item["height_mm"] = payload["height_mm"]
    item["length_mm"] = payload["height_mm"]
    item["area_mm2"] = payload["area_mm2"]
    item.setdefault("debug", {})
    item["debug"].update({
        "rotated_width_px": payload["width_px"],
        "rotated_height_px": payload["height_px"],
        "rotated_angle": payload["angle"],
    })
    return item


def _wrinkle_quality(box, circle, shape, pixels_per_mm):
    if not circle or not box:
        return "review_calibration"
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    if x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2:
        return "review_segmentation"
    bounds = _circle_bounds(circle, shape, 0.08)
    if not bounds:
        return "review_calibration"
    if _intersect_box(box, bounds) != box:
        return "review_segmentation"
    if _mask_overlap_ratio(box, circle, shape, 0.98) < 0.50:
        return "review_segmentation"
    if _box_center_distance_ratio(box, circle) > 0.58:
        return "review_segmentation"
    w_mm, h_mm, area = _box_mm(box, pixels_per_mm)
    if min(w_mm, h_mm) < 14 or max(w_mm, h_mm) > 112 or area < 420 or area > 9000:
        return "review_segmentation"
    return "ok"


def _inside_ratio(box, circle):
    if not circle:
        return 0.0
    cx, cy, radius, _ = circle
    x1, y1, x2, y2 = box
    pts = [(x1, y1), (x2, y1), (x1, y2), (x2, y2), ((x1 + x2) / 2.0, (y1 + y2) / 2.0)]
    inside = 0
    limit = radius * 0.98
    for x, y in pts:
        if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= limit:
            inside += 1
    return inside / float(len(pts))


def _mask_overlap_ratio(box, circle, shape, shrink=0.96):
    if not box or not circle:
        return 0.0
    clipped = _clip_box(box, shape)
    if not clipped:
        return 0.0
    x1, y1, x2, y2 = clipped
    area = max(1, (x2 - x1) * (y2 - y1))
    mask = _circle_mask(shape, circle, shrink)
    return float(np.count_nonzero(mask[y1:y2, x1:x2])) / float(area)


def _box_center_distance_ratio(box, circle):
    if not box or not circle:
        return 99.0
    cx, cy, radius, _ = circle
    bx = (box[0] + box[2]) / 2.0
    by = (box[1] + box[3]) / 2.0
    return (((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5) / max(radius, 1.0)


def _overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / float(area)


def _quality_flag(class_name, box, circle, shape=None, pixels_per_mm=0.0):
    if not circle or pixels_per_mm <= 0:
        return "review_calibration"
    if class_name == "zhe_zhou" and shape is not None:
        return _wrinkle_quality(box, circle, shape, pixels_per_mm)
    w_mm, h_mm, area = _box_mm(box, pixels_per_mm)
    if class_name == "zhe_zhou":
        if area < 3000 or area > 9000 or min(w_mm, h_mm) < 45 or max(w_mm, h_mm) > 110:
            return "review_segmentation"
        return "ok"
    if class_name == "zang_wu":
        if shape is not None:
            if _mask_overlap_ratio(box, circle, shape, 0.98) < 0.55 and not _box_center_inside(box, circle, 0.92):
                return "review_segmentation"
        elif _inside_ratio(box, circle) < 0.45:
            return "review_segmentation"
        if area < 60 or area > 620 or min(w_mm, h_mm) < 4:
            return "review_segmentation"
        return "ok"
    if _inside_ratio(box, circle) < 0.75:
        return "review_segmentation"
    return "ok"

def _make_item(class_name, box, score, prediction_time=None, mode="refined_component"):
    x1, y1, x2, y2 = [int(v) for v in box]
    item = {
        "class_name": class_name,
        "score": round(float(score), 4),
        "loc": [x1, y1, x2, y2],
        "refined": True,
        "refine_mode": mode,
        "quality_flag": "ok",
    }
    if class_name == "zang_wu" and "dark" in mode:
        item["decision_source"] = "dark_component"
    elif class_name == "zhe_zhou" and "wrinkle" in mode:
        item["decision_source"] = "wrinkle_cross"
    else:
        item["decision_source"] = "model_fallback" if "fallback" in mode else mode
    if prediction_time is not None:
        item["prediction_time"] = prediction_time
    return item


def find_sheet_status(image, cf=None):
    gray = _gray(image)
    circle = _find_sheet(gray, cf)
    if gray is None or not circle:
        return {"present": False, "confidence": 0.0}
    cx, cy, radius, _ = circle
    center_roi = _center_roi_from_config(cf, gray.shape)
    center_dist = _center_distance_to_roi(cx, cy, center_roi)
    max_center_dist = _max_center_distance_from_config(cf, radius)
    if center_dist > max_center_dist:
        return {
            "present": False,
            "confidence": 0.0,
            "cx": round(float(cx), 2),
            "cy": round(float(cy), 2),
            "radius": round(float(radius), 2),
            "reject_reason": "invalid_roi",
            "target_distance_px": round(float(center_dist), 2),
        }
    min_radius, max_radius, expected_radius = _sheet_radius_range_from_config(cf)
    radius_conf = 0.7
    if min_radius > 0 and max_radius > 0:
        if min_radius <= radius <= max_radius:
            radius_conf = 1.0
        else:
            nearest = min(abs(radius - min_radius), abs(radius - max_radius))
            tolerance = max((max_radius - min_radius) * 0.35, 1.0)
            radius_conf = max(0.0, 1.0 - nearest / tolerance)
    mask = _circle_mask(gray.shape, circle, 0.92)
    vals = gray[mask > 0]
    texture_conf = 0.5
    if vals.size > 50:
        texture_conf = min(1.0, max(0.2, (float(np.std(vals)) + float(np.mean(vals)) / 18.0) / 22.0))
    confidence = round(float(max(0.0, min(1.0, 0.65 * radius_conf + 0.35 * texture_conf))), 3)
    return {
        "present": confidence >= 0.35,
        "confidence": confidence,
        "cx": round(float(cx), 2),
        "cy": round(float(cy), 2),
        "radius": round(float(radius), 2),
    }


def _box_inside_circle_score(box, circle):
    if not box or not circle:
        return 0.0
    cx, cy, radius, _ = circle
    x1, y1, x2, y2 = box
    pts = [
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
    ]
    score = 0.0
    for x, y in pts:
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if dist <= radius * 1.05:
            score += 1.0
    return score / float(len(pts))


def _refine_sheet_with_model_boxes(gray, cf, circle, result):
    boxes = [_bbox(item) for item in (result or []) if item.get("class_name") in ("zhe_zhou", "zang_wu")]
    boxes = [box for box in boxes if box]
    if gray is None or not boxes:
        return circle
    current_score = max([_box_inside_circle_score(box, circle) for box in boxes] or [0.0]) if circle else 0.0
    if current_score >= 0.55:
        return circle
    if circle and _wrinkle_box(gray, circle) is not None:
        return circle
    min_radius, max_radius, expected_radius = _sheet_radius_range_from_config(cf)
    candidates = _hough_sheet_candidates(gray, min_radius, max_radius, expected_radius)
    if circle:
        candidates.append(circle)
    best = circle
    best_score = current_score
    for cand in candidates:
        cx, cy, radius, _ = cand
        radius_err = abs(radius - expected_radius) / max(expected_radius, 1.0) if expected_radius > 0 else 0.0
        score = max([_box_inside_circle_score(box, cand) for box in boxes] or [0.0]) - radius_err * 0.15
        # On the contest camera the disk should be well inside the frame; avoid
        # selecting belt/fixture circles that do not explain the model box.
        if score >= 0.70 and score > best_score + 0.08:
            best_score = score
            best = cand
    return best



def _sheet_circle_for_scoring(gray, cf, circle):
    """Prefer the calibrated 100mm disk radius when a false outer circle wins."""
    if gray is None or not circle:
        return circle
    ppm = _pixels_per_mm_from_config(cf)
    if ppm <= 0:
        return circle
    try:
        ref_mm = float(cf.get("Measurement", "calibration_diameter_mm", fallback="100") or 100)
    except Exception:
        ref_mm = 100.0
    if ref_mm <= 0:
        return circle
    expected = ppm * ref_mm / 2.0
    cx, cy, radius, contour = circle
    radius_err = abs(float(radius) - expected) / max(expected, 1.0)
    if radius_err <= 0.08:
        return circle
    min_radius, max_radius, _ = _sheet_radius_range_from_config(cf)
    candidates = _hough_sheet_candidates(gray, min_radius, max_radius, expected)
    if circle:
        candidates.append(circle)
    best = circle
    best_score = 1e9
    h, w = gray.shape[:2]
    frame_cx, frame_cy = w / 2.0, h / 2.0
    center_roi = _center_roi_from_config(cf, gray.shape)
    for cand in candidates:
        ccx, ccy, rr, cont = cand
        if rr <= 0:
            continue
        err = abs(float(rr) - expected) / max(expected, 1.0)
        roi_dist = _center_distance_to_roi(ccx, ccy, center_roi)
        roi_limit = _max_center_distance_from_config(cf, rr)
        if roi_dist > roi_limit * 1.25:
            continue
        edge_penalty = 0.0
        if ccx - rr < 0 or ccy - rr < 0 or ccx + rr >= w or ccy + rr >= h:
            edge_penalty += 0.35
        center_penalty = math.hypot(ccx - frame_cx, ccy - frame_cy) / max(max(w, h), 1)
        roi_penalty = roi_dist / max(roi_limit, 1.0)
        score = err + center_penalty * 0.10 + roi_penalty * 0.20 + edge_penalty
        if score < best_score:
            best_score = score
            best = cand
    return best


def _expand_box(box, dx, dy, shape):
    x1, y1, x2, y2 = box
    return _clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), shape) or box



def _adjust_dirty_scoring_box(box, circle, shape, pixels_per_mm):
    # The contest asks for the minimum rectangle covering the whole dirty mark,
    # not only the darkest core. Keep a small calibrated guard band for soft ink
    # edges and camera blur.
    if not circle:
        return _expand_box(box, 3, 3, shape)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    base = max(3, int(round(max(pixels_per_mm, 0.1) * 1.25)))
    extra = max(1, int(round(max(pixels_per_mm, 0.1) * 0.55)))
    if w < h * 0.72:
        pad_x, pad_y = base + extra, base
    elif h < w * 0.72:
        pad_x, pad_y = base, base + extra
    else:
        pad_x = pad_y = base
    return _expand_box(box, pad_x, pad_y, shape)

def _shrink_dark_box(gray, box, circle, residual=None):
    # Backward-compatible name; this now grows from the strict dark seed to the
    # full soft boundary of the same dirty mark. The previous strict-core box was
    # visibly smaller than the stains and under-measured width/height.
    x1, y1, x2, y2 = box
    if residual is None:
        bg = cv2.GaussianBlur(gray, (0, 0), 25)
        residual = cv2.subtract(bg, gray)
    if not circle:
        return _expand_box(box, 4, 4, gray.shape)

    _, _, radius, _ = circle
    search_pad = max(6, int(round(radius * 0.055)))
    ex = _expand_box(box, search_pad, search_pad, gray.shape)
    x1e, y1e, x2e, y2e = ex
    patch = gray[y1e:y2e, x1e:x2e]
    rpatch = residual[y1e:y2e, x1e:x2e]
    if patch.size == 0 or rpatch.size == 0:
        return box

    sheet_mask = _circle_mask(gray.shape, circle, 0.92)
    mpatch = sheet_mask[y1e:y2e, x1e:x2e] > 0
    disk_vals = gray[sheet_mask > 0]
    residual_vals = residual[sheet_mask > 0]
    if disk_vals.size < 50 or residual_vals.size < 50:
        return box

    disk_median = float(np.median(disk_vals))
    strong_limit = max(15.0, float(np.percentile(residual_vals, 90)))
    soft_limit = max(8.0, min(strong_limit * 0.72, float(np.percentile(residual_vals, 82))))
    local_dark_limit = min(disk_median + 5.0, float(np.percentile(patch[mpatch], 68)) if np.any(mpatch) else disk_median + 5.0)

    soft = ((rpatch >= soft_limit) & (patch <= local_dark_limit) & mpatch).astype('uint8') * 255
    kernel = np.ones((3, 3), np.uint8)
    soft = cv2.morphologyEx(soft, cv2.MORPH_CLOSE, kernel, iterations=1)

    seed = np.zeros_like(soft, dtype=np.uint8)
    sx1 = max(0, x1 - x1e)
    sy1 = max(0, y1 - y1e)
    sx2 = min(seed.shape[1], x2 - x1e)
    sy2 = min(seed.shape[0], y2 - y1e)
    if sx2 <= sx1 or sy2 <= sy1:
        return box
    seed[sy1:sy2, sx1:sx2] = 255

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(soft, 8)
    keep = []
    for label in range(1, labels_count):
        component = labels == label
        if not np.any(component & (seed > 0)):
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        keep.append(component)
    if not keep:
        return box

    union = np.zeros_like(soft, dtype=bool)
    for component in keep:
        union |= component
    ys, xs = np.where(union)
    if xs.size == 0 or ys.size == 0:
        return box
    candidate = (x1e + int(xs.min()), y1e + int(ys.min()), x1e + int(xs.max()) + 1, y1e + int(ys.max()) + 1)

    if _inside_ratio(candidate, circle) < 0.72:
        return box
    cw, ch = candidate[2] - candidate[0], candidate[3] - candidate[1]
    if cw < 5 or ch < 5:
        return box
    if cw > radius * 0.52 or ch > radius * 0.52:
        return box
    return candidate




def _box_area(box):
    if not box:
        return 0
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _merge_boxes(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _box_center_inside(box, circle, shrink=0.86):
    if not box or not circle:
        return False
    cx, cy, radius, _ = circle
    bx = (box[0] + box[2]) / 2.0
    by = (box[1] + box[3]) / 2.0
    return ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 <= radius * shrink


def _box_center_in_box(inner, outer, pad=0):
    if not inner or not outer:
        return False
    bx = (inner[0] + inner[2]) / 2.0
    by = (inner[1] + inner[3]) / 2.0
    return (outer[0] - pad) <= bx <= (outer[2] + pad) and (outer[1] - pad) <= by <= (outer[3] + pad)


def _dirty_boxes_same_stain(a, b, circle):
    if not a or not b or not circle:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
    bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
    x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    x_ratio = x_overlap / float(max(1, min(aw, bw)))
    y_ratio = y_overlap / float(max(1, min(ah, bh)))
    vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
    horizontal_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    _, _, radius, _ = circle
    # Split masks often cut one tall stain into top/bottom components. Merge
    # those when they share the same x-band and are separated by only a small
    # lighting gap. Keep distant independent stains separated.
    if x_ratio >= 0.42 and vertical_gap <= max(10, int(radius * 0.12)):
        return True
    if y_ratio >= 0.42 and horizontal_gap <= max(8, int(radius * 0.08)):
        return True
    return False


def _dedupe_dirty_items(items, circle):
    merged = []
    for item in sorted(items, key=lambda it: (_bbox(it)[1], _bbox(it)[0]) if _bbox(it) else (0, 0)):
        box = _bbox(item)
        if not box:
            continue
        # Real stains can sit near the sheet edge, so a scoring rectangle may
        # have one or two corners outside the inscribed circle. Keep it when
        # the defect center is still on the sheet; otherwise edge stains get
        # dropped and "three spots" becomes one visible box.
        if _inside_ratio(box, circle) < 0.70 and not _box_center_inside(box, circle, 1.02):
            continue
        matched = False
        for old in merged:
            old_box = _bbox(old)
            if not old_box:
                continue
            if _overlap_ratio(box, old_box) > 0.55 or _overlap_ratio(old_box, box) > 0.55 or _dirty_boxes_same_stain(box, old_box, circle):
                new_box = _merge_boxes(box, old_box)
                old['loc'] = [int(v) for v in new_box]
                old['score'] = max(float(old.get('score', 0)), float(item.get('score', 0)))
                old['measurement_rect_mode'] = 'dirty_merged_min_rect'
                old.pop('rotated_rect_px', None)
                old.pop('box_width_mm', None)
                old.pop('box_height_mm', None)
                old.pop('width_mm', None)
                old.pop('height_mm', None)
                old.pop('length_mm', None)
                old.pop('area_mm2', None)
                matched = True
                break
        if not matched:
            merged.append(item)
    merged.sort(key=lambda it: (_bbox(it)[1], _bbox(it)[0]) if _bbox(it) else (0, 0))
    return merged[:6]

def _dirty_texture_ok(gray, box, circle, residual, residual_limit):
    if not box or not circle:
        return True
    x1, y1, x2, y2 = box
    patch = gray[y1:y2, x1:x2]
    res_patch = residual[y1:y2, x1:x2]
    if patch.size == 0 or res_patch.size == 0:
        return False
    cx, cy, radius, _ = circle
    bx = (x1 + x2) / 2.0
    by = (y1 + y2) / 2.0
    radial = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / max(radius, 1.0)
    strong_ratio = float(np.mean(res_patch >= residual_limit))
    strong_mean = float(np.mean(res_patch))
    # Lighting gradients and rim shadows can be dark, but they do not have the
    # compact residual core of a real dirty mark.
    if strong_ratio < 0.24 or strong_mean < residual_limit * 1.2:
        return False
    return True

def _wrinkle_box(gray, circle):
    if not circle:
        return None
    mask = _circle_mask(gray.shape, circle, 0.88)
    if not np.any(mask):
        return None
    cx, cy, radius, _ = circle
    vals = gray[mask > 0]
    if vals.size < 50:
        return None

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blur)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge_limit = max(float(np.percentile(mag[mask > 0], 88)), 18.0)
    edges = ((mag >= edge_limit) & (mask > 0)).astype('uint8') * 255
    canny = cv2.Canny(eq, 40, 110)
    edges = cv2.bitwise_or(edges, cv2.bitwise_and(canny, mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    hough_inputs = [edges]
    bg = cv2.GaussianBlur(gray, (0, 0), max(9, int(radius * 0.10)))
    residual = cv2.subtract(bg, gray)
    ridge_limit = max(12.0, float(np.percentile(residual[mask > 0], 88)))
    ridge = ((residual >= ridge_limit) & (mask > 0)).astype("uint8") * 255
    ridge = cv2.morphologyEx(ridge, cv2.MORPH_CLOSE, kernel, iterations=1)
    hough_inputs.append(ridge)

    lines_all = []
    for hough_image in hough_inputs:
        lines = cv2.HoughLinesP(
            hough_image,
            rho=1,
            theta=np.pi / 180,
            threshold=max(14, int(radius * 0.13)),
            minLineLength=max(24, int(radius * 0.30)),
            maxLineGap=max(8, int(radius * 0.12)),
        )
        if lines is not None:
            lines_all.extend(lines[:, 0, :].tolist())
    if not lines_all:
        return None

    candidates = []
    for raw in lines_all:
        x1, y1, x2, y2 = [int(v) for v in raw]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5 > radius * 0.78:
            continue
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < radius * 0.34:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        candidates.append((x1, y1, x2, y2, angle, length))
    if len(candidates) < 2:
        return None

    best_pair = None
    best_score = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            diff = abs(a[4] - b[4])
            diff = min(diff, 180.0 - diff)
            if diff < 35.0 or diff > 145.0:
                continue
            pts = np.array([[a[0], a[1]], [a[2], a[3]], [b[0], b[1]], [b[2], b[3]]], dtype=np.float32)
            center_dist = float(np.mean(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)))
            score = a[5] + b[5] - center_dist * 0.35 + diff
            if score > best_score:
                best_score = score
                best_pair = (a, b)
    if best_pair is None:
        return None

    pts = []
    for line in best_pair:
        pts.extend([(line[0], line[1]), (line[2], line[3])])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = max(8, int(radius * 0.11))
    merged = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    bounds = _circle_bounds(circle, gray.shape, 0.02)
    clipped = _intersect_box(merged, bounds)
    if not clipped:
        clipped = _clip_box(merged, gray.shape)
    if not clipped:
        return None
    x1c, y1c, x2c, y2c = clipped
    local = cv2.bitwise_or(edges, ridge)[y1c:y2c, x1c:x2c]
    local_mask = mask[y1c:y2c, x1c:x2c]
    line_mask = _line_distance_mask(gray.shape, best_pair, max(5.0, radius * 0.045))[y1c:y2c, x1c:x2c]
    active = (local > 0) & (local_mask > 0) & line_mask
    if int(np.count_nonzero(active)) < max(55, int(radius * 1.05)):
        return None
    if np.any(active):
        ay, ax = np.where(active)
        tight = (
            x1c + int(ax.min()) - max(3, int(radius * 0.025)),
            y1c + int(ay.min()) - max(3, int(radius * 0.025)),
            x1c + int(ax.max()) + 1 + max(3, int(radius * 0.025)),
            y1c + int(ay.max()) + 1 + max(3, int(radius * 0.025)),
        )
        clipped = _intersect_box(tight, bounds) or clipped
    clipped = _expand_wrinkle_box_to_cross(clipped, circle, gray.shape)
    w, h = clipped[2] - clipped[0], clipped[3] - clipped[1]
    if w < radius * 0.34 or h < radius * 0.34:
        return None
    if w > radius * 1.75 or h > radius * 1.75:
        return None
    if _mask_overlap_ratio(clipped, circle, gray.shape, 0.98) < 0.45:
        return None
    if _box_center_distance_ratio(clipped, circle) > 0.42:
        return None
    return clipped


def _weak_wrinkle_box(gray, circle):
    if not circle:
        return None
    mask = _circle_mask(gray.shape, circle, 0.88)
    if not np.any(mask):
        return None
    cx, cy, radius, _ = circle
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge_limit = max(20.0, float(np.percentile(mag[mask > 0], 84)))
    edges = ((mag >= edge_limit) & (mask > 0)).astype("uint8") * 255
    canny = cv2.Canny(eq, 32, 95)
    edges = cv2.bitwise_or(edges, cv2.bitwise_and(canny, mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, int(radius * 0.11)),
        minLineLength=max(24, int(radius * 0.28)),
        maxLineGap=max(8, int(radius * 0.12)),
    )
    if lines is None:
        return None
    groups = []
    for raw in lines[:, 0, :].tolist():
        x1, y1, x2, y2 = [int(v) for v in raw]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5 > radius * 0.72:
            continue
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < radius * 0.28:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        groups.append((x1, y1, x2, y2, angle, length))
    if len(groups) < 2:
        return None
    best_pair = None
    best_score = -1.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            diff = abs(a[4] - b[4])
            diff = min(diff, 180.0 - diff)
            if diff < 35.0 or diff > 145.0:
                continue
            pts = np.array([[a[0], a[1]], [a[2], a[3]], [b[0], b[1]], [b[2], b[3]]], dtype=np.float32)
            center_dist = float(np.mean(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)))
            if center_dist > radius * 0.66:
                continue
            score = a[5] + b[5] + diff - center_dist * 0.42
            if score > best_score:
                best_score = score
                best_pair = (a, b)
    if best_pair is None:
        return None
    pts = []
    for line in best_pair:
        pts.extend([(line[0], line[1]), (line[2], line[3])])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = max(7, int(radius * 0.08))
    bounds = _circle_bounds(circle, gray.shape, 0.02)
    clipped = _intersect_box((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad), bounds)
    if not clipped:
        return None
    x1c, y1c, x2c, y2c = clipped
    line_mask = _line_distance_mask(gray.shape, best_pair, max(5.0, radius * 0.045))[y1c:y2c, x1c:x2c]
    local_edges = edges[y1c:y2c, x1c:x2c]
    local_mask = mask[y1c:y2c, x1c:x2c]
    active = (local_edges > 0) & (local_mask > 0) & line_mask
    if int(np.count_nonzero(active)) < max(45, int(radius * 0.85)):
        return None
    if np.any(active):
        ay, ax = np.where(active)
        pad_tight = max(3, int(radius * 0.025))
        clipped = _intersect_box((
            x1c + int(ax.min()) - pad_tight,
            y1c + int(ay.min()) - pad_tight,
            x1c + int(ax.max()) + 1 + pad_tight,
            y1c + int(ay.max()) + 1 + pad_tight,
        ), bounds) or clipped
    clipped = _expand_wrinkle_box_to_cross(clipped, circle, gray.shape)
    w, h = clipped[2] - clipped[0], clipped[3] - clipped[1]
    if w < radius * 0.32 or h < radius * 0.32:
        return None
    if w > radius * 1.75 or h > radius * 1.75:
        return None
    if _mask_overlap_ratio(clipped, circle, gray.shape, 0.98) < 0.55:
        return None
    if _box_center_distance_ratio(clipped, circle) > 0.40:
        return None
    return clipped


def _is_wrinkle_scene(gray, circle, result):
    # Do not trust the SSD wrinkle label by itself: dirty and partial-sheet
    # frames often produce coarse wrinkle boxes. The final decision requires
    # a real two-line crossing inside the sheet.
    return _wrinkle_box(gray, circle) is not None or _weak_wrinkle_box(gray, circle) is not None

def _detect_dark_components(gray, circle, model_boxes, base_score, prediction_time, pixels_per_mm):
    mask = _circle_mask(gray.shape, circle, 0.84)
    if not np.any(mask):
        return []
    vals = gray[mask > 0]
    if vals.size < 50:
        return []
    med = float(np.median(vals))
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    residual = cv2.subtract(bg, gray)
    residual_vals = residual[mask > 0]
    if residual_vals.size < 50:
        return []
    cx, cy, radius, _ = circle
    limit = max(10.0, float(np.percentile(residual_vals, 86)))

    # Use three independent cues so the same stain count remains stable under
    # changing glare: local residual, absolute darkness, and black-hat spots.
    residual_dark = (residual > limit) & (mask > 0)
    abs_limit = min(med - 8.0, float(np.percentile(vals, 27)))
    abs_dark = (gray <= abs_limit) & (mask > 0)
    kernel_size = max(9, int(radius * 0.13))
    if kernel_size % 2 == 0:
        kernel_size += 1
    spot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, spot_kernel)
    bh_vals = blackhat[mask > 0]
    bh_limit = max(7.0, float(np.percentile(bh_vals, 84))) if bh_vals.size else 7.0
    blackhat_dark = (blackhat >= bh_limit) & (mask > 0)
    support = (residual_dark | abs_dark | blackhat_dark).astype("uint8") * 255

    kernel = np.ones((3, 3), np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, kernel, iterations=1)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel, iterations=1)

    seed_limit = min(med - 18.0, float(np.percentile(vals, 9)))
    seed = ((gray <= seed_limit) & (mask > 0)).astype("uint8") * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel, iterations=1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel, iterations=1)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    min_dim = max(5, int(radius * 0.04))
    max_dim = radius * 0.50
    items = []
    seen = []
    for label in range(1, labels_count):
        seed_area = int(stats[label, cv2.CC_STAT_AREA])
        sx = int(stats[label, cv2.CC_STAT_LEFT])
        sy = int(stats[label, cv2.CC_STAT_TOP])
        sw = int(stats[label, cv2.CC_STAT_WIDTH])
        sh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if seed_area < max(18, int(radius * radius * 0.0012)) or sw < 3 or sh < 3:
            continue
        seed_aspect = max(sw, sh) / float(max(1, min(sw, sh)))
        if seed_aspect > 3.2 or sw > max_dim or sh > max_dim:
            continue

        search_pad = max(7, int(radius * 0.075))
        search = _expand_box((sx, sy, sx + sw, sy + sh), search_pad, search_pad, gray.shape)
        if not search:
            continue
        x1s, y1s, x2s, y2s = search
        roi_gray = gray[y1s:y2s, x1s:x2s]
        roi_res = residual[y1s:y2s, x1s:x2s]
        roi_support = support[y1s:y2s, x1s:x2s] > 0
        roi_mask = mask[y1s:y2s, x1s:x2s] > 0
        roi_seed = (labels[y1s:y2s, x1s:x2s] == label)
        if roi_gray.size == 0 or not np.any(roi_seed):
            continue

        seed_vals = roi_gray[roi_seed]
        seed_med = float(np.median(seed_vals)) if seed_vals.size else seed_limit
        grow_limit = min(med - 5.0, seed_med + 24.0)
        grow = (((roi_gray <= grow_limit) | (roi_res >= max(6.0, limit * 0.45)) | roi_support) & roi_mask).astype("uint8") * 255
        grow = cv2.morphologyEx(grow, cv2.MORPH_CLOSE, kernel, iterations=1)
        grow_labels_count, grow_labels, grow_stats, _ = cv2.connectedComponentsWithStats(grow, 8)
        union = np.zeros_like(grow, dtype=bool)
        for grow_label in range(1, grow_labels_count):
            component = grow_labels == grow_label
            if np.any(component & roi_seed):
                union |= component
        if not np.any(union):
            union = roi_seed
        ys, xs = np.where(union)
        if xs.size == 0 or ys.size == 0:
            continue
        x, y = x1s + int(xs.min()), y1s + int(ys.min())
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        area = float(np.count_nonzero(union))
        if area < max(35, radius * radius * 0.002) or w < min_dim or h < min_dim:
            continue
        if w > max_dim or h > max_dim:
            continue
        box = (x, y, x + w, y + h)
        if _inside_ratio(box, circle) < 0.78:
            continue
        extent = float(area) / float(max(1, w * h))
        aspect = max(w, h) / float(max(1, min(w, h)))
        if extent < 0.26 or aspect > 2.9:
            continue
        patch = gray[y:y + h, x:x + w]
        res_patch = residual[y:y + h, x:x + w]
        if patch.size == 0 or res_patch.size == 0:
            continue
        local_union = union[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
        dark_pixels = patch[local_union]
        if dark_pixels.size < max(15, int(area * 0.22)):
            continue
        if float(np.mean(dark_pixels)) > med - 5.0:
            continue
        if float(np.mean(res_patch)) < limit * 0.42 and float(np.mean(dark_pixels)) > med - 12.0:
            continue
        if float(np.mean(patch)) > med + 4 and float(np.mean(res_patch)) < limit * 1.2:
            continue
        if model_boxes and max(_overlap_ratio(box, mb) for mb in model_boxes) < 0.003 and area < 70:
            continue
        duplicate = False
        for old in seen:
            if _overlap_ratio(box, old) > 0.42 or _overlap_ratio(old, box) > 0.42:
                duplicate = True
                break
        if duplicate:
            continue
        component_mask = np.zeros_like(union, dtype=np.uint8)
        component_mask[union] = 255
        rect_payload = _rotated_rect_payload(
            _rotated_rect_from_mask(component_mask, offset=(x1s, y1s)),
            gray.shape,
            pixels_per_mm,
        )
        box = _adjust_dirty_scoring_box(box, circle, gray.shape, pixels_per_mm)
        if not _dirty_texture_ok(gray, box, circle, residual, limit):
            continue
        if rect_payload and rect_payload.get("bbox") and _inside_ratio(rect_payload["bbox"], circle) >= 0.70:
            box = rect_payload["bbox"]
        if box in seen:
            continue
        seen.append(box)
        score = max(0.58, min(0.99, base_score - 0.02 + min(area / 1400.0, 0.18)))
        item = _make_item("zang_wu", box, score, prediction_time, "measured_dark_component")
        if rect_payload and rect_payload.get("bbox"):
            item = _apply_measurement_rect(item, rect_payload, "dirty_min_area_rect")
            item["component_id"] = len(items) + 1
        item["quality_flag"] = _quality_flag("zang_wu", box, circle, gray.shape, pixels_per_mm)
        items.append(item)
    items = _dedupe_dirty_items(items, circle)
    items.sort(key=lambda it: (it["loc"][2] - it["loc"][0]) * (it["loc"][3] - it["loc"][1]), reverse=True)
    return items[:6]


def _detect_dark_components(gray, circle, model_boxes, base_score, prediction_time, pixels_per_mm):
    """Contest scoring dirty detector: one independent dark stain -> one box."""
    mask = _circle_mask(gray.shape, circle, 0.88)
    if not np.any(mask):
        return []
    vals = gray[mask > 0]
    if vals.size < 80:
        return []
    cx, cy, radius, _ = circle
    med = float(np.median(vals))
    bg = cv2.GaussianBlur(gray, (0, 0), max(17, int(radius * 0.18)))
    residual = cv2.subtract(bg, gray)
    residual_vals = residual[mask > 0]
    if residual_vals.size < 80:
        return []

    r82 = float(np.percentile(residual_vals, 82))
    r88 = float(np.percentile(residual_vals, 88))
    dark_abs = float(np.percentile(vals, 30))
    strong_abs = float(np.percentile(vals, 14))
    kernel_size = max(11, int(radius * 0.18))
    if kernel_size % 2 == 0:
        kernel_size += 1
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    bh_vals = blackhat[mask > 0]
    bh80 = float(np.percentile(bh_vals, 80)) if bh_vals.size else 7.0

    support = (
        ((residual >= max(6.0, r82 * 0.78)) & (gray <= min(med + 8.0, dark_abs + 18.0))) |
        ((residual >= max(8.0, r88 * 0.60)) & (blackhat >= max(5.0, bh80 * 0.58))) |
        ((gray <= min(med - 8.0, strong_abs + 10.0)) & (blackhat >= max(4.0, bh80 * 0.45)))
    ) & (mask > 0)

    # YOLO boxes are candidates, not scoring boxes. They only relax local
    # thresholding so a weak third stain is still split and measured.
    for mb in model_boxes or []:
        mb = _clip_box(mb, gray.shape)
        if not mb:
            continue
        x1, y1, x2, y2 = _expand_box(mb, max(4, int(radius * 0.035)), max(4, int(radius * 0.035)), gray.shape)
        local_mask = mask[y1:y2, x1:x2] > 0
        if not np.any(local_mask):
            continue
        patch = gray[y1:y2, x1:x2]
        rpatch = residual[y1:y2, x1:x2]
        local_dark = float(np.percentile(patch[local_mask], 48))
        local_res = float(np.percentile(rpatch[local_mask], 68))
        support[y1:y2, x1:x2] |= ((patch <= min(med + 10.0, local_dark + 10.0)) & (rpatch >= max(4.0, local_res * 0.72)) & local_mask)

    support = support.astype("uint8") * 255
    k3 = np.ones((3, 3), np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, k3, iterations=1)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, k3, iterations=2)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)

    min_area_px = max(28, int(radius * radius * 0.0010))
    min_dim = max(4, int(radius * 0.028))
    max_dim = radius * 0.54
    items = []
    seen = []
    for label in range(1, labels_count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area_px < min_area_px or w < min_dim or h < min_dim:
            continue
        if w > max_dim or h > max_dim:
            continue
        aspect = max(w, h) / float(max(1, min(w, h)))
        if aspect > 3.4:
            continue
        box = (x, y, x + w, y + h)
        if _mask_overlap_ratio(box, circle, gray.shape, 0.96) < 0.58:
            continue
        component = (labels[y:y + h, x:x + w] == label)
        patch = gray[y:y + h, x:x + w]
        rpatch = residual[y:y + h, x:x + w]
        if patch.size == 0 or not np.any(component):
            continue
        comp_vals = patch[component]
        comp_res = rpatch[component]
        if comp_vals.size < max(12, int(area_px * 0.25)):
            continue
        if float(np.mean(comp_vals)) > med + 2.0 and float(np.mean(comp_res)) < max(5.0, r82 * 0.55):
            continue
        extent = float(area_px) / float(max(1, w * h))
        if extent < 0.16:
            continue
        if not _dirty_texture_ok(gray, box, circle, residual, max(7.0, r82)):
            # Keep model-supported weak spots, but reject unsupported texture.
            if not model_boxes or max(_overlap_ratio(box, mb) for mb in model_boxes) < 0.05:
                continue
        duplicate = False
        for old in seen:
            if _overlap_ratio(box, old) > 0.36 or _overlap_ratio(old, box) > 0.36:
                duplicate = True
                break
        if duplicate:
            continue
        component_mask = np.zeros((h, w), dtype=np.uint8)
        component_mask[component] = 255
        component_mask = cv2.dilate(component_mask, k3, iterations=1)
        rect_payload = _rotated_rect_payload(_rotated_rect_from_mask(component_mask, offset=(x, y), min_points=5), gray.shape, pixels_per_mm)
        grown_box = _adjust_dirty_scoring_box(box, circle, gray.shape, pixels_per_mm)
        if rect_payload and rect_payload.get("bbox") and _mask_overlap_ratio(rect_payload["bbox"], circle, gray.shape, 0.98) >= 0.58:
            final_box = rect_payload["bbox"]
        else:
            final_box = grown_box
            rect_payload = _rect_payload_from_box(final_box, gray.shape, pixels_per_mm)
        if not final_box:
            continue
        seen.append(final_box)
        score = max(0.62, min(0.99, float(base_score or 0.78) - 0.03 + min(area_px / 1800.0, 0.20)))
        item = _make_item("zang_wu", final_box, score, prediction_time, "measured_dark_component")
        if rect_payload:
            item = _apply_measurement_rect(item, rect_payload, "dirty_min_area_rect")
        item["component_id"] = len(items) + 1
        item["quality_flag"] = _quality_flag("zang_wu", final_box, circle, gray.shape, pixels_per_mm)
        item.setdefault("debug", {})
        item["debug"].update({
            "component_area_px": area_px,
            "disk_median": round(med, 2),
            "residual_p82": round(r82, 2),
        })
        items.append(item)

    items = _dedupe_dirty_items(items, circle)
    items.sort(key=lambda it: (it["loc"][1], it["loc"][0]))
    for idx, item in enumerate(items, 1):
        item["component_id"] = idx
    return items[:8]


def _add_missing_model_dirty_items(gray, circle, items, model_boxes, model_scores, prediction_time, pixels_per_mm):
    items = list(items or [])
    existing = [_bbox(item) for item in items if _bbox(item)]
    for idx, box in enumerate(model_boxes or []):
        score = float(model_scores[idx]) if idx < len(model_scores or []) else 0.72
        if score < 0.35:
            continue
        clipped = _clip_box(box, gray.shape)
        if not clipped or _mask_overlap_ratio(clipped, circle, gray.shape, 0.98) < 0.50:
            continue
        if not _model_dirty_box_has_dark_evidence(gray, circle, clipped, min_pixels=18, min_ratio=0.012):
            continue
        if existing and max(_overlap_ratio(clipped, old) for old in existing) > 0.24:
            continue
        x1, y1, x2, y2 = _expand_box(clipped, max(2, int(pixels_per_mm * 0.9)), max(2, int(pixels_per_mm * 0.9)), gray.shape)
        sheet_mask = _circle_mask(gray.shape, circle, 0.90)
        patch = gray[y1:y2, x1:x2]
        roi_mask = sheet_mask[y1:y2, x1:x2] > 0
        if patch.size == 0 or not np.any(roi_mask):
            continue
        vals = gray[sheet_mask > 0]
        med = float(np.median(vals)) if vals.size else float(np.median(patch))
        bg = cv2.GaussianBlur(gray, (0, 0), 25)
        residual = cv2.subtract(bg, gray)
        rpatch = residual[y1:y2, x1:x2]
        local_limit = min(med + 8.0, float(np.percentile(patch[roi_mask], 56)))
        res_limit = max(5.0, float(np.percentile(rpatch[roi_mask], 55)))
        comp = ((patch <= local_limit) & (rpatch >= res_limit) & roi_mask).astype("uint8") * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(comp, 8)
        best_label = 0
        best_area = 0
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_label = label
        if best_label > 0 and best_area >= max(18, int((x2 - x1) * (y2 - y1) * 0.08)):
            component_mask = (labels == best_label).astype("uint8") * 255
            component_mask = cv2.dilate(component_mask, np.ones((3, 3), np.uint8), iterations=1)
            rect_payload = _rotated_rect_payload(_rotated_rect_from_mask(component_mask, offset=(x1, y1), min_points=5), gray.shape, pixels_per_mm)
        else:
            rect_payload = _rect_payload_from_box(clipped, gray.shape, pixels_per_mm)
        if not rect_payload or not rect_payload.get("bbox"):
            continue
        final_box = rect_payload["bbox"]
        if _mask_overlap_ratio(final_box, circle, gray.shape, 0.98) < 0.45:
            continue
        item = _make_item("zang_wu", final_box, max(0.58, min(0.97, score)), prediction_time, "measured_dark_component")
        item = _apply_measurement_rect(item, rect_payload, "dirty_min_area_rect")
        item["quality_flag"] = _quality_flag("zang_wu", final_box, circle, gray.shape, pixels_per_mm)
        item.setdefault("debug", {})
        item["debug"].update({"model_fill": True, "model_score": round(score, 4)})
        items.append(item)
        existing.append(final_box)
    items = _dedupe_dirty_items(items, circle)
    items.sort(key=lambda it: (it["loc"][1], it["loc"][0]))
    for idx, item in enumerate(items, 1):
        item["component_id"] = idx
    return items


def _model_dirty_box_has_dark_evidence(gray, circle, box, min_pixels=22, min_ratio=0.018):
    if gray is None or not circle or not box:
        return False
    try:
        clipped = _clip_box(box, gray.shape)
        if not clipped:
            return False
        x1, y1, x2, y2 = clipped
        sheet_mask = _circle_mask(gray.shape, circle, 0.90)
        patch = gray[y1:y2, x1:x2]
        roi_mask = sheet_mask[y1:y2, x1:x2] > 0
        roi_count = int(np.count_nonzero(roi_mask))
        if patch.size == 0 or roi_count < patch.size * 0.35:
            return False
        vals = gray[sheet_mask > 0]
        if vals.size < 100:
            return False
        med = float(np.median(vals))
        p25 = float(np.percentile(vals, 25))
        bg = cv2.GaussianBlur(gray, (0, 0), max(15, int((circle[2] or 80) * 0.16)))
        residual = cv2.subtract(bg, gray)
        rpatch = residual[y1:y2, x1:x2]
        residual_vals = residual[sheet_mask > 0]
        r88 = float(np.percentile(residual_vals, 88)) if residual_vals.size else 0.0
        dark = (patch < min(med - 12.0, p25 - 5.0)) & roi_mask
        strong = (rpatch > max(12.0, r88 * 0.85)) & roi_mask
        evidence = dark & strong
        evidence_count = int(np.count_nonzero(evidence))
        if evidence_count < max(int(min_pixels), int(roi_count * min_ratio)):
            return False
        ys, xs = np.where(evidence)
        if xs.size == 0 or ys.size == 0:
            return False
        ew = int(xs.max() - xs.min() + 1)
        eh = int(ys.max() - ys.min() + 1)
        if ew < 4 or eh < 4:
            return False
        return True
    except Exception:
        return False


def _align_dirty_items_to_model_boxes(gray, circle, items, model_boxes, model_scores, prediction_time, pixels_per_mm):
    """Use high-confidence YOLO dirty boxes as the full stain envelope.

    The local dark mask is intentionally conservative so it does not eat
    wrinkle texture, but for contest scoring that can under-box a real black
    stain. When YOLO confidently separates dirty spots, keep one scoring box per
    model spot and use the model envelope clipped to the sheet as the minimum
    rectangle floor.
    """
    if not model_boxes:
        return items or []
    by_model = []
    used_items = set()
    multi_model_dirty = len(model_boxes or []) >= 3
    for idx, raw_box in enumerate(model_boxes):
        payload = None
        score = float(model_scores[idx]) if idx < len(model_scores or []) else 0.0
        if score < 0.50:
            continue
        model_box = _clip_box(tuple(int(round(v)) for v in raw_box), gray.shape)
        if not model_box or _mask_overlap_ratio(model_box, circle, gray.shape, 0.98) < 0.45:
            continue
        strong_yolo_split = multi_model_dirty and score >= 0.78
        if not strong_yolo_split and not _model_dirty_box_has_dark_evidence(gray, circle, model_box, min_pixels=18, min_ratio=0.012):
            continue
        best_i = -1
        best_score = -1.0
        for item_i, item in enumerate(items or []):
            if item_i in used_items:
                continue
            box = _bbox(item)
            if not box:
                continue
            overlap = max(_overlap_ratio(box, model_box), _overlap_ratio(model_box, box))
            center_bonus = 0.35 if _box_center_in_box(box, model_box, pad=4) else 0.0
            rank = overlap + center_bonus
            if rank > best_score:
                best_score = rank
                best_i = item_i
        base_box = _bbox(items[best_i]) if best_i >= 0 and best_score > 0.02 else None
        if best_i >= 0 and best_score > 0.02:
            used_items.add(best_i)
        if base_box is None:
            # A model-only dirty box is only a seed. Contest scoring must come
            # from actual dark evidence inside the sheet, not an empty model box.
            x1, y1, x2, y2 = model_box
            sheet_mask = _circle_mask(gray.shape, circle, 0.92)
            patch = gray[y1:y2, x1:x2]
            roi_mask = sheet_mask[y1:y2, x1:x2] > 0
            if patch.size == 0 or not np.any(roi_mask):
                continue
            bg = cv2.GaussianBlur(gray, (0, 0), 21)
            residual = cv2.subtract(bg, gray)
            rpatch = residual[y1:y2, x1:x2]
            local_dark = float(np.percentile(patch[roi_mask], 45))
            local_res = float(np.percentile(rpatch[roi_mask], 62))
            support = ((patch <= local_dark + 6.0) & (rpatch >= max(4.0, local_res * 0.70)) & roi_mask).astype("uint8") * 255
            support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            support_area = int(np.count_nonzero(support))
            fill = support_area / float(max(1, (x2 - x1) * (y2 - y1)))
            if support_area < max(18, int(circle[2] * circle[2] * 0.0007)) or fill < 0.055:
                if strong_yolo_split:
                    payload = _rect_payload_from_box(model_box, gray.shape, pixels_per_mm)
                    final_box = model_box
                else:
                    continue
            else:
                payload = _rotated_rect_payload(_rotated_rect_from_mask(support, offset=(x1, y1), min_points=5), gray.shape, pixels_per_mm)
                if not payload or not payload.get("bbox"):
                    if strong_yolo_split:
                        payload = _rect_payload_from_box(model_box, gray.shape, pixels_per_mm)
                        final_box = model_box
                    else:
                        continue
                else:
                    final_box = payload["bbox"]
            if not payload or not payload.get("bbox"):
                continue
        else:
            final_box = _merge_boxes(base_box, model_box)
            # A local dark component may leak into a highlight boundary on
            # brushed aluminum. For high-confidence, separated dirty spots the
            # model box is a reliable per-stain envelope; keep the scoring
            # rectangle near that envelope instead of unioning unrelated glare.
            try:
                bx1, by1, bx2, by2 = base_box
                mx1, my1, mx2, my2 = model_box
                bw = max(1, bx2 - bx1)
                bh = max(1, by2 - by1)
                mw = max(1, mx2 - mx1)
                mh = max(1, my2 - my1)
                leak_ratio = max(float(bw) / mw, float(bh) / mh)
                if score >= 0.72 and leak_ratio > 1.28:
                    margin = max(2, int(round(max(pixels_per_mm, 1.0) * 0.8)))
                    envelope = _expand_box(model_box, margin, margin, gray.shape)
                    constrained = _intersect_box(base_box, envelope)
                    if constrained:
                        final_box = constrained
                    else:
                        final_box = model_box
                    payload = _rect_payload_from_box(final_box, gray.shape, pixels_per_mm)
            except Exception:
                pass
        final_box = _intersect_box(final_box, _circle_bounds(circle, gray.shape, 0.01)) or _clip_box(final_box, gray.shape)
        if not final_box:
            continue
        w = final_box[2] - final_box[0]
        h = final_box[3] - final_box[1]
        _, _, radius, _ = circle
        if w <= 3 or h <= 3 or w > radius * 0.66 or h > radius * 0.66:
            continue
        payload = payload if payload and payload.get("bbox") == final_box else _rect_payload_from_box(final_box, gray.shape, pixels_per_mm)
        if not payload:
            continue
        item = _make_item("zang_wu", final_box, max(0.70, min(0.99, score)), prediction_time, "measured_dark_component")
        item = _apply_measurement_rect(item, payload, "dirty_model_guided_min_area_rect")
        item["quality_flag"] = _quality_flag("zang_wu", final_box, circle, gray.shape, pixels_per_mm)
        item.setdefault("debug", {})
        item["debug"].update({
            "model_guided": True,
            "model_score": round(score, 4),
            "model_box": [int(v) for v in model_box],
            "local_overlap_rank": round(best_score, 3),
        })
        by_model.append(item)
    if len(by_model) < 2:
        for item_i, item in enumerate(items or []):
            if item_i in used_items:
                continue
            box = _bbox(item)
            if not box:
                continue
            if by_model and max(max(_overlap_ratio(box, _bbox(other)), _overlap_ratio(_bbox(other), box)) for other in by_model if _bbox(other)) > 0.18:
                continue
            by_model.append(item)
    by_model = _dedupe_dirty_items(by_model, circle)
    by_model.sort(key=lambda it: (it["loc"][1], it["loc"][0]))
    for idx, item in enumerate(by_model, 1):
        item["component_id"] = idx
    return by_model


def _force_model_dirty_scoring_items(gray, circle, items, model_boxes, model_scores, prediction_time, pixels_per_mm):
    """Keep one contest scoring box for each high-confidence YOLO dirty spot.

    Local dark segmentation is still preferred, but the competition rule says
    each independent stain needs its own minimum rectangle. When YOLO separates
    stains with high confidence, missing one is worse than using the model
    envelope as a conservative scoring rectangle.
    """
    merged = list(items or [])
    existing = [_bbox(item) for item in merged if _bbox(item)]
    multi_model_dirty = len(model_boxes or []) >= 3
    for idx, raw_box in enumerate(model_boxes or []):
        score = float(model_scores[idx]) if idx < len(model_scores or []) else 0.0
        if score < (0.78 if multi_model_dirty else 0.82):
            continue
        model_box = _clip_box(tuple(int(round(v)) for v in raw_box), gray.shape)
        if not model_box:
            continue
        overlap = _mask_overlap_ratio(model_box, circle, gray.shape, 0.98)
        if overlap < 0.42:
            continue
        if score < 0.90 and overlap < 0.55:
            continue
        strong_yolo_split = multi_model_dirty and score >= 0.78
        if not strong_yolo_split and not _model_dirty_box_has_dark_evidence(gray, circle, model_box, min_pixels=18, min_ratio=0.012):
            continue
        if existing and max(max(_overlap_ratio(model_box, old), _overlap_ratio(old, model_box)) for old in existing) > 0.30:
            continue
        x1, y1, x2, y2 = model_box
        if (x2 - x1) <= 4 or (y2 - y1) <= 4:
            continue
        if (x2 - x1) > gray.shape[1] * 0.28 or (y2 - y1) > gray.shape[0] * 0.32:
            continue
        payload = _rect_payload_from_box(model_box, gray.shape, pixels_per_mm)
        if not payload:
            continue
        item = _make_item("zang_wu", model_box, max(0.88, min(0.99, score)), prediction_time, "measured_dark_component")
        item = _apply_measurement_rect(item, payload, "dirty_yolo_forced_min_area_rect")
        item["quality_flag"] = "ok" if score >= 0.90 else _quality_flag("zang_wu", model_box, circle, gray.shape, pixels_per_mm)
        item.setdefault("debug", {})
        item["debug"].update({
            "model_forced": True,
            "model_forced_reason": "yolo_split_candidate" if strong_yolo_split else "dark_evidence",
            "model_score": round(score, 4),
            "model_box": [int(v) for v in model_box],
            "circle_overlap": round(float(overlap), 3),
        })
        merged.append(item)
        existing.append(model_box)
    merged.sort(key=lambda it: (it["loc"][1], it["loc"][0]))
    for idx, item in enumerate(merged, 1):
        item["component_id"] = idx
    return merged[:8]


def _dark_components_are_confident_dirty(items, pixels_per_mm):
    """Reject wrinkle texture fragments that only look like small dark spots."""
    if not items or pixels_per_mm <= 0:
        return False
    areas = []
    ok_count = 0
    for item in items:
        box = _bbox(item)
        if not box:
            continue
        _, _, area = _box_mm(box, pixels_per_mm)
        areas.append(area)
        if item.get("quality_flag") == "ok":
            ok_count += 1
    if not areas:
        return False
    if len(areas) == 1:
        return ok_count >= 1 and 70.0 <= float(areas[0]) <= 900.0
    median_area = float(np.median(areas))
    max_area = float(max(areas))
    # Real dirty sample after calibration is usually around 140-210 mm2 per
    # spot. Allow slightly smaller real-world stains, but reject wrinkle center
    # fragments, which are usually three tiny boxes below 110 mm2.
    if len(areas) >= 3:
        # A dirty sample in this contest can have multiple separate dark stains.
        # If three measured dark components are present, keep them as dirty even
        # when the SSD coarse classifier says wrinkle; wrinkle fragments are much
        # smaller and fail these area floors.
        return median_area >= 60.0 and max_area >= 90.0 and max_area <= 1200.0
    return ok_count >= 1 and median_area >= 70.0 and max_area >= 90.0 and max_area <= 1200.0


def _dark_components_are_probable_dirty(items, pixels_per_mm):
    """Softer multi-spot guard used before wrinkle fallback.

    Multiple compact dark components are a stronger contest cue for dirty than
    weak wrinkle lines. Keep the floor below the final confident threshold so
    dark stains are not swallowed by the wrinkle fallback, but still reject the
    tiny fragments produced around wrinkle intersections.
    """
    if not items or pixels_per_mm <= 0:
        return False
    if len(items) < 2:
        return False
    areas = []
    for item in items:
        box = _bbox(item)
        if not box:
            continue
        _, _, area = _box_mm(box, pixels_per_mm)
        areas.append(float(area))
    if len(areas) < 2:
        return False
    median_area = float(np.median(areas))
    max_area = float(max(areas))
    return median_area >= 38.0 and max_area >= 65.0 and max_area <= 1400.0


def _valid_wrinkle(gray, item, circle):
    box = _bbox(item)
    if not box or not circle or _inside_ratio(box, circle) < 0.8:
        return False
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    cx, cy, radius, _ = circle
    if w > radius * 0.75 or h > radius * 0.75:
        return False
    ratio = max(w, h) / float(max(1, min(w, h)))
    if ratio < 2.0:
        return False
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(cv2.magnitude(gx, gy)))
    return edge >= 18.0


def _line_distance_mask(shape, lines, max_distance):
    h, w = shape[:2]
    yy, xx = np.indices((h, w))
    mask = np.zeros((h, w), dtype=bool)
    for line in lines:
        x1, y1, x2, y2 = [float(v) for v in line[:4]]
        dx = x2 - x1
        dy = y2 - y1
        denom = math.hypot(dx, dy)
        if denom <= 1e-6:
            continue
        dist = np.abs(dy * xx - dx * yy + x2 * y1 - y2 * x1) / denom
        along = ((xx - x1) * dx + (yy - y1) * dy) / (denom * denom)
        mask |= (dist <= max_distance) & (along >= -0.10) & (along <= 1.10)
    return mask


def _line_sample_stats(gray, dark_residual, bright_residual, glare_mask, line):
    try:
        x1, y1, x2, y2 = [float(v) for v in line[:4]]
        length = max(2, int(round(math.hypot(x2 - x1, y2 - y1))))
        xs = np.linspace(x1, x2, length).astype(np.int32)
        ys = np.linspace(y1, y2, length).astype(np.int32)
        h, w = gray.shape[:2]
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        dark_mean = float(np.mean(dark_residual[ys, xs])) if length else 0.0
        bright_mean = float(np.mean(bright_residual[ys, xs])) if length else 0.0
        gray_mean = float(np.mean(gray[ys, xs])) if length else 0.0
        glare_ratio = float(np.mean(glare_mask[ys, xs] > 0)) if length else 0.0
        return dark_mean, bright_mean, gray_mean, glare_ratio
    except Exception:
        return 0.0, 0.0, 0.0, 1.0


def _point_to_segment_distance(px, py, line):
    try:
        x1, y1, x2, y2 = [float(v) for v in line[:4]]
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        if denom <= 1e-6:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
        qx = x1 + t * dx
        qy = y1 + t * dy
        return math.hypot(px - qx, py - qy)
    except Exception:
        return 1e9


def _line_intersection(a, b):
    try:
        x1, y1, x2, y2 = [float(v) for v in a[:4]]
        x3, y3, x4, y4 = [float(v) for v in b[:4]]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return px, py
    except Exception:
        return None


def _scale_rotated_rect(rect, factor, max_side=None):
    try:
        (cx, cy), (rw, rh), angle = rect
        rw = float(rw)
        rh = float(rh)
        factor = float(factor)
        if max_side and max(rw, rh) * factor > float(max_side):
            factor = float(max_side) / max(rw, rh)
        factor = max(1.0, factor)
        return ((float(cx), float(cy)), (rw * factor, rh * factor), float(angle))
    except Exception:
        return rect


def _clip_line_around_point(line, px, py, max_half_length):
    try:
        x1, y1, x2, y2 = [float(v) for v in line[:4]]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return line
        ux = dx / length
        uy = dy / length
        t1 = (x1 - px) * ux + (y1 - py) * uy
        t2 = (x2 - px) * ux + (y2 - py) * uy
        lo = max(min(t1, t2), -float(max_half_length))
        hi = min(max(t1, t2), float(max_half_length))
        if hi - lo < max(12.0, float(max_half_length) * 0.35):
            lo = -float(max_half_length) * 0.55
            hi = float(max_half_length) * 0.55
        return (
            px + ux * lo,
            py + uy * lo,
            px + ux * hi,
            py + uy * hi,
        ) + tuple(line[4:])
    except Exception:
        return line


def _expand_wrinkle_box_to_cross(box, circle, shape):
    if not box or not circle:
        return box
    cx, cy, radius, _ = circle
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    min_span = max(18, int(round(radius * 0.42)))
    max_span = max(28, int(round(radius * 1.12)))
    if w >= min_span and h >= min_span:
        return box
    half_x = min(max(w / 2.0, min_span / 2.0), max_span / 2.0)
    half_y = min(max(h / 2.0, min_span / 2.0), max_span / 2.0)
    ex = (x1 + x2) / 2.0
    ey = (y1 + y2) / 2.0
    if ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5 > radius * 0.42:
        ex = cx * 0.45 + ex * 0.55
        ey = cy * 0.45 + ey * 0.55
    center_box = (
        int(round(ex - half_x)),
        int(round(ey - half_y)),
        int(round(ex + half_x)),
        int(round(ey + half_y)),
    )
    merged = _merge_boxes(box, center_box)
    bounds = _circle_bounds(circle, shape, 0.01)
    return _intersect_box(merged, bounds) or _clip_box(merged, shape) or box


def _wrinkle_union_rect(gray, circle, seed_box=None, pixels_per_mm=0.0):
    if gray is None or not circle:
        return None, None
    cx, cy, radius, _ = circle
    mask = _circle_mask(gray.shape, circle, 0.91)
    if not np.any(mask):
        return None, None
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[mask > 0]
    if vals.size < 50:
        return None, None
    edge_limit = max(18.0, float(np.percentile(vals, 84)))
    edges = ((mag >= edge_limit) & (mask > 0)).astype("uint8") * 255
    canny = cv2.Canny(eq, 32, 105)
    edges = cv2.bitwise_or(edges, cv2.bitwise_and(canny, mask))
    bg = cv2.GaussianBlur(gray, (0, 0), max(9, int(radius * 0.10)))
    dark_residual = cv2.subtract(bg, gray)
    bright_residual = cv2.subtract(gray, bg)
    dark_vals = dark_residual[mask > 0]
    bright_vals = bright_residual[mask > 0]
    dark = ((dark_residual >= max(10.0, float(np.percentile(dark_vals, 86)))) & (mask > 0)).astype("uint8") * 255
    bright = ((bright_residual >= max(10.0, float(np.percentile(bright_vals, 88)))) & (mask > 0)).astype("uint8") * 255
    glare_limit = max(205.0, float(np.percentile(gray[mask > 0], 92)) if np.any(mask) else 235.0)
    glare = ((gray >= glare_limit) & (mask > 0)).astype("uint8") * 255
    evidence = cv2.bitwise_or(edges, cv2.bitwise_or(dark, bright))
    evidence = cv2.bitwise_and(evidence, cv2.bitwise_not(cv2.erode(glare, np.ones((5, 5), np.uint8), iterations=1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        evidence,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, int(radius * 0.10)),
        minLineLength=max(24, int(radius * 0.25)),
        maxLineGap=max(8, int(radius * 0.13)),
    )
    if lines is None:
        return None, None
    candidates = []
    for raw in lines[:, 0, :].tolist():
        x1, y1, x2, y2 = [int(v) for v in raw]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5 > radius * 0.86:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        if length < radius * 0.24:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        dark_mean, bright_mean, gray_mean, glare_ratio = _line_sample_stats(gray, dark_residual, bright_residual, glare, (x1, y1, x2, y2))
        if glare_ratio > 0.42 and dark_mean < bright_mean * 0.70:
            continue
        # Wrinkle folds are usually thin dark/bright transitions. The saturated
        # highlight boundary is strong but not a scoring defect, so keep it weak.
        fold_score = length + dark_mean * 1.9 + bright_mean * 0.35 - glare_ratio * radius * 1.2
        candidates.append((x1, y1, x2, y2, angle, length, fold_score, dark_mean, bright_mean, glare_ratio))
    if len(candidates) < 2:
        return None, None
    best = None
    best_score = -1e9
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            diff = abs(a[4] - b[4])
            diff = min(diff, 180.0 - diff)
            if diff < 30.0 or diff > 150.0:
                continue
            intersection = _line_intersection(a, b)
            if intersection is None:
                continue
            ix, iy = intersection
            if ((ix - cx) ** 2 + (iy - cy) ** 2) ** 0.5 > radius * 0.48:
                continue
            if _point_to_segment_distance(ix, iy, a) > radius * 0.16:
                continue
            if _point_to_segment_distance(ix, iy, b) > radius * 0.16:
                continue
            pts = np.array([[a[0], a[1]], [a[2], a[3]], [b[0], b[1]], [b[2], b[3]]], dtype=np.float32)
            center_dist = float(np.mean(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)))
            pair_box = _merge_boxes((a[0], a[1], a[2], a[3]), (b[0], b[1], b[2], b[3]))
            pair_center = _box_center_distance_ratio(pair_box, circle) if pair_box else 1.0
            intersection_dist = math.hypot(ix - cx, iy - cy)
            glare_penalty = (float(a[9]) + float(b[9])) * radius * 0.85
            score = a[6] + b[6] + diff * 1.8 - center_dist * 0.34 - pair_center * radius * 0.55 - intersection_dist * 0.18 - glare_penalty
            if seed_box:
                merged = pair_box
                if merged and (_overlap_ratio(seed_box, merged) > 0 or _overlap_ratio(merged, seed_box) > 0):
                    score += radius * 0.08
            if score > best_score:
                best_score = score
                best = (a, b)
    if best is None:
        return None, None
    a, b = best
    angle_diff = abs(float(a[4]) - float(b[4]))
    angle_diff = min(angle_diff, 180.0 - angle_diff)
    local_strength = (
        (float(a[7]) + float(b[7])) * 1.35
        + (float(a[8]) + float(b[8])) * 0.28
        + (float(a[5]) + float(b[5])) / max(float(radius), 1.0) * 12.0
        + angle_diff * 0.22
        - (float(a[9]) + float(b[9])) * 38.0
    )
    cross = _line_intersection(best[0], best[1])
    if cross is None:
        return None, None
    # Score the crossed wrinkle body around the intersection. Long Hough lines
    # can ride along the right-side specular highlight; clipping near the cross
    # preserves the contest target while avoiding that highlight tail.
    half_len = max(48.0, radius * 0.58)
    best = (
        _clip_line_around_point(best[0], cross[0], cross[1], half_len),
        _clip_line_around_point(best[1], cross[0], cross[1], half_len),
    )
    line_mask = _line_distance_mask(gray.shape, best, max(5.0, radius * 0.060))
    active = ((evidence > 0) & (mask > 0) & line_mask).astype("uint8") * 255
    active = cv2.dilate(active, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    # Scoring requires one rectangle covering the crossed wrinkle pair. Sparse
    # edge pixels often under-cover the visible folds, so add the two selected
    # line segments themselves and their endpoints before minAreaRect.
    stroke = np.zeros_like(active)
    for line in best:
        x1, y1, x2, y2 = [int(round(v)) for v in line[:4]]
        cv2.line(stroke, (x1, y1), (x2, y2), 255, max(3, int(radius * 0.045)))
    active = cv2.bitwise_or(active, cv2.bitwise_and(stroke, mask))
    # The model ROI is only a class/location hint. Do not union raw evidence
    # from the whole model box here: on the current lighting, the right-side
    # saturated highlight edge is stronger than the actual fold and pulls the
    # scoring rectangle away from the crossed wrinkles.
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    if labels_count <= 1:
        return None, None
    keep = np.zeros_like(active, dtype=np.uint8)
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(25, int(radius * 0.24)):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        box = (x, y, x + w, y + h)
        if seed_box and _overlap_ratio(box, seed_box) <= 0 and _overlap_ratio(seed_box, box) <= 0 and _box_center_distance_ratio(box, circle) > 0.64:
            continue
        keep[labels == label] = 255
    if int(np.count_nonzero(keep)) < max(55, int(radius * 0.75)):
        return None, None
    union_points = []
    for line in best:
        union_points.append([float(line[0]), float(line[1])])
        union_points.append([float(line[2]), float(line[3])])
    # The model seed must not dictate the scoring rectangle corners. The final
    # rectangle is built from fold lines plus local evidence only.
    pts = cv2.findNonZero(keep)
    if pts is not None:
        local_points = pts.reshape(-1, 2).astype(float)
        # Keep evidence near the selected fold strokes. This removes specular
        # highlight texture that is inside the disk but not part of the wrinkle.
        d0 = _line_distance_mask(gray.shape, (best[0],), max(7.0, radius * 0.050))
        d1 = _line_distance_mask(gray.shape, (best[1],), max(7.0, radius * 0.050))
        near = (d0 | d1)
        filtered = []
        for px, py in local_points.tolist():
            ix, iy = int(round(px)), int(round(py))
            if 0 <= iy < near.shape[0] and 0 <= ix < near.shape[1] and near[iy, ix]:
                if math.hypot(px - cross[0], py - cross[1]) <= radius * 0.68:
                    filtered.append([px, py])
        union_points.extend(filtered)
    if len(union_points) >= 6:
        rect = cv2.minAreaRect(np.array(union_points, dtype=np.float32))
    else:
        rect = _rotated_rect_from_mask(keep)
    if rect is not None:
        # Scoring requires the minimum rectangle covering the two crossed folds,
        # not the earlier presentation square. Keep only a small margin for
        # edge-thickness/segmentation noise.
        rect = _scale_rotated_rect(rect, 1.04, max_side=radius * 1.12)
    payload = _rotated_rect_payload(rect, gray.shape, pixels_per_mm)
    if not payload or not payload.get("bbox"):
        return None, None
    bbox = payload["bbox"]
    expanded_bbox = _expand_wrinkle_box_to_cross(bbox, circle, gray.shape)
    if expanded_bbox:
        ew = expanded_bbox[2] - expanded_bbox[0]
        eh = expanded_bbox[3] - expanded_bbox[1]
        center_ok = _box_center_distance_ratio(expanded_bbox, circle) <= 0.58
        overlap_ok = _mask_overlap_ratio(expanded_bbox, circle, gray.shape, 0.98) >= 0.55
        if ew <= radius * 1.05 and eh <= radius * 1.05 and center_ok and overlap_ok:
            payload["bbox"] = expanded_bbox
            payload.setdefault("debug", {})
            payload["debug"]["display_bbox_expanded"] = True
    if _mask_overlap_ratio(payload["bbox"], circle, gray.shape, 0.98) < 0.40:
        return None, None
    if _box_center_distance_ratio(payload["bbox"], circle) > 0.62:
        return None, None
    if payload is not None:
        payload.setdefault("debug", {})
        payload["debug"].update({
            "local_cross_strength": round(float(local_strength), 3),
            "angle_diff": round(float(angle_diff), 2),
            "line_lengths": [round(float(a[5]), 1), round(float(b[5]), 1)],
            "dark_means": [round(float(a[7]), 2), round(float(b[7]), 2)],
            "bright_means": [round(float(a[8]), 2), round(float(b[8]), 2)],
            "glare_ratios": [round(float(a[9]), 3), round(float(b[9]), 3)],
            "has_model_seed": bool(seed_box),
            "wrinkle_rect_rule": "fold_union_min_area_rect",
        })
    scoring_payload = _competition_wrinkle_payload(cross, circle, gray.shape, pixels_per_mm, payload, "local_cross", seed_box)
    if scoring_payload and scoring_payload.get("bbox"):
        return scoring_payload["bbox"], scoring_payload
    return payload["bbox"], payload


def _wrinkle_model_scoring_payload(gray, circle, seed_box, pixels_per_mm=0.0):
    if gray is None or not circle or not seed_box:
        return None, None
    cx, cy, radius, _ = circle
    x1, y1, x2, y2 = seed_box
    sx = max(x1, min(x2, cx))
    sy = max(y1, min(y2, cy))
    if math.hypot(float(sx) - float(cx), float(sy) - float(cy)) > radius * 0.28:
        sx, sy = cx, cy
    payload = _competition_wrinkle_payload((sx, sy), circle, gray.shape, pixels_per_mm, None, "model_seed", seed_box)
    if not payload or not payload.get("bbox"):
        return None, None
    if _mask_overlap_ratio(payload["bbox"], circle, gray.shape, 0.98) < 0.50:
        return None, None
    payload.setdefault("debug", {})
    payload["debug"].update({
        "model_seed_bbox": [int(v) for v in seed_box],
        "wrinkle_rect_rule": "competition_cross_scoring_rect",
    })
    return payload["bbox"], payload


def _wrinkle_payload_strong_enough(payload, model_score, has_model_seed=False):
    """Prevent normal glare from becoming NG when the model gives no support."""
    try:
        model_score = float(model_score or 0.0)
    except Exception:
        model_score = 0.0
    if model_score >= 0.45:
        return True
    if model_score >= 0.32 and has_model_seed:
        return True
    debug = payload.get("debug", {}) if isinstance(payload, dict) else {}
    try:
        strength = float(debug.get("local_cross_strength", 0) or 0)
        angle_diff = float(debug.get("angle_diff", 0) or 0)
        glare_ratios = [float(v) for v in (debug.get("glare_ratios") or [])]
        line_lengths = [float(v) for v in (debug.get("line_lengths") or [])]
        dark_means = [float(v) for v in (debug.get("dark_means") or [])]
    except Exception:
        return False
    glare_avg = sum(glare_ratios) / max(len(glare_ratios), 1)
    min_len = min(line_lengths) if line_lengths else 0.0
    dark_sum = sum(dark_means)
    # Local-only wrinkle must be a real crossed fold, not a single specular edge.
    return (
        strength >= 58.0
        and 38.0 <= angle_diff <= 132.0
        and glare_avg <= 0.34
        and min_len >= 54.0
        and dark_sum >= 16.0
    )


def refine_defect_output(image, out_put, cf=None):
    if not isinstance(out_put, dict):
        return out_put
    result = out_put.get("result") or []
    gray = _gray(image)
    if gray is None:
        return out_put
    circle = _find_sheet(gray, cf)
    circle = _refine_sheet_with_model_boxes(gray, cf, circle, result)
    circle = _sheet_circle_for_scoring(gray, cf, circle)
    if gray is None or not circle:
        # A model box without a reliable sheet circle cannot be scored by the
        # contest rules. Treat it as no formal defect instead of letting a
        # glare-induced SSD wrinkle box turn a normal or off-center sheet into
        # NG.
        out_put = dict(out_put)
        out_put["result"] = []
        out_put["len"] = 0
        out_put["refine_status"] = "no_sheet_ignore_model"
        out_put["decision_source"] = "sheet_unreliable"
        return out_put
    pixels_per_mm = _pixels_per_mm_from_config(cf)
    if pixels_per_mm <= 0:
        out_put["refine_status"] = "not_calibrated_fallback"
        return out_put

    normal = []
    z_boxes = []
    z_scores = []
    pred_time = None
    for item in result:
        if pred_time is None and item.get("prediction_time") is not None:
            pred_time = item.get("prediction_time")
    model_z_boxes = []
    model_z_scores = []
    model_wrinkle_boxes = []
    model_wrinkle_scores = []
    for item in result:
        if item.get("class_name", "") == "zhe_zhou":
            box = _bbox(item)
            clipped = _clip_box(box, gray.shape) if box else None
            if clipped and _inside_ratio(clipped, circle) > 0.25:
                model_wrinkle_boxes.append(clipped)
                model_wrinkle_scores.append(_round_score(item.get("score", 0.0)))
            continue
        if item.get("class_name", "") != "zang_wu":
            continue
        box = _bbox(item)
        if not box:
            continue
        clipped = _clip_box(box, gray.shape)
        score = _round_score(item.get("score", 0.75))
        # Strong YOLO dirty boxes are useful seeds even when the sheet circle is
        # slightly mislocalized by glare/wrinkles. Keep them so the contest
        # "one stain, one rectangle" rule is not broken by a bad mask.
        if clipped and (_inside_ratio(clipped, circle) > 0.12 or score >= 0.82):
            model_z_boxes.append(clipped)
            model_z_scores.append(score)
    model_wrinkle_score = max(model_wrinkle_scores or [0.0])
    wrinkle_seed = max(model_wrinkle_boxes, key=_box_area) if model_wrinkle_boxes else None
    if model_wrinkle_score >= 0.55 and wrinkle_seed:
        wrinkle_box, wrinkle_payload = _wrinkle_union_rect(gray, circle, wrinkle_seed, pixels_per_mm)
        if not wrinkle_box and model_wrinkle_score >= 0.70:
            wrinkle_box, wrinkle_payload = _wrinkle_model_scoring_payload(gray, circle, wrinkle_seed, pixels_per_mm)
        if wrinkle_box and _wrinkle_payload_strong_enough(wrinkle_payload, model_wrinkle_score, True):
            out_put = dict(out_put)
            item = _make_item("zhe_zhou", wrinkle_box, max(model_wrinkle_score, 0.82), pred_time, "measured_wrinkle_cross")
            if wrinkle_payload:
                item = _apply_measurement_rect(item, wrinkle_payload, "wrinkle_competition_cross_rect")
            item["quality_flag"] = _quality_flag("zhe_zhou", wrinkle_box, circle, gray.shape, pixels_per_mm)
            out_put["result"] = [item]
            out_put["len"] = 1
            out_put["refine_status"] = "refined_wrinkle_model_priority"
            out_put["decision_source"] = "wrinkle_cross"
            return out_put
    early_dark = _detect_dark_components(gray, circle, [], 0.82, pred_time, pixels_per_mm)
    confident_dirty = _dark_components_are_confident_dirty(early_dark, pixels_per_mm)
    multi_model_dirty = len(model_z_boxes) >= 2 and max(model_z_scores or [0.0]) >= 0.50
    if multi_model_dirty:
        model_dark = _detect_dark_components(gray, circle, model_z_boxes, max(model_z_scores or [0.78]), pred_time, pixels_per_mm)
        if len(model_dark) < len(model_z_boxes):
            model_dark = _add_missing_model_dirty_items(gray, circle, model_dark, model_z_boxes, model_z_scores, pred_time, pixels_per_mm)
        model_dark = _align_dirty_items_to_model_boxes(
            gray, circle, model_dark, model_z_boxes, model_z_scores, pred_time, pixels_per_mm
        )
        model_dark = _force_model_dirty_scoring_items(
            gray, circle, model_dark, model_z_boxes, model_z_scores, pred_time, pixels_per_mm
        )
        if len(model_dark) >= 2 and (_dark_components_are_confident_dirty(model_dark, pixels_per_mm) or _dark_components_are_probable_dirty(model_dark, pixels_per_mm)):
            out_put = dict(out_put)
            out_put["result"] = model_dark
            out_put["len"] = len(model_dark)
            out_put["refine_status"] = "refined_model_dark_components_priority"
            out_put["decision_source"] = "dark_component"
            return out_put
    if len(early_dark) >= 1 and confident_dirty and len(early_dark) >= max(1, len(model_z_boxes)) and not multi_model_dirty:
        out_put = dict(out_put)
        out_put["result"] = early_dark
        out_put["len"] = len(early_dark)
        out_put["refine_status"] = "refined_dark_components_priority"
        out_put["decision_source"] = "dark_component"
        return out_put
    if len(early_dark) >= 2 and _dark_components_are_probable_dirty(early_dark, pixels_per_mm) and len(early_dark) >= max(2, len(model_z_boxes)) and not multi_model_dirty:
        out_put = dict(out_put)
        out_put["result"] = early_dark
        out_put["len"] = len(early_dark)
        out_put["refine_status"] = "refined_dark_components_multi_priority"
        out_put["decision_source"] = "dark_component"
        return out_put

    if model_z_boxes:
        model_dark = _detect_dark_components(gray, circle, model_z_boxes, max(model_z_scores or [0.78]), pred_time, pixels_per_mm)
        if len(model_dark) < len(model_z_boxes):
            model_dark = _add_missing_model_dirty_items(gray, circle, model_dark, model_z_boxes, model_z_scores, pred_time, pixels_per_mm)
        model_dark = _align_dirty_items_to_model_boxes(
            gray, circle, model_dark, model_z_boxes, model_z_scores, pred_time, pixels_per_mm
        )
        model_dark = _force_model_dirty_scoring_items(
            gray, circle, model_dark, model_z_boxes, model_z_scores, pred_time, pixels_per_mm
        )
        if model_dark and _dark_components_are_confident_dirty(model_dark, pixels_per_mm):
            out_put = dict(out_put)
            out_put["result"] = model_dark
            out_put["len"] = len(model_dark)
            out_put["refine_status"] = "refined_model_dark_components_priority"
            out_put["decision_source"] = "dark_component"
            return out_put
        if len(model_dark) >= 2 and _dark_components_are_probable_dirty(model_dark, pixels_per_mm):
            out_put = dict(out_put)
            out_put["result"] = model_dark
            out_put["len"] = len(model_dark)
            out_put["refine_status"] = "refined_model_dark_components_multi_priority"
            out_put["decision_source"] = "dark_component"
            return out_put

    wrinkle_box, wrinkle_payload = _wrinkle_union_rect(gray, circle, wrinkle_seed, pixels_per_mm)
    if not wrinkle_box and model_wrinkle_score >= 0.70 and wrinkle_seed:
        wrinkle_box, wrinkle_payload = _wrinkle_model_scoring_payload(gray, circle, wrinkle_seed, pixels_per_mm)
    if wrinkle_box and not _wrinkle_payload_strong_enough(wrinkle_payload, model_wrinkle_score, bool(wrinkle_seed)):
        wrinkle_box = None
        wrinkle_payload = None
    if not wrinkle_box and model_wrinkle_score >= 0.45:
        wrinkle_box = _wrinkle_box(gray, circle) or _weak_wrinkle_box(gray, circle)
        wrinkle_payload = None
    wrinkle_scene = wrinkle_box is not None
    if wrinkle_scene:
        box = wrinkle_box
        if box:
            out_put = dict(out_put)
            wrinkle_score = max(model_wrinkle_score, 0.82 if wrinkle_payload else 0.62)
            item = _make_item("zhe_zhou", box, wrinkle_score, pred_time, "measured_wrinkle_cross")
            if wrinkle_payload:
                item = _apply_measurement_rect(item, wrinkle_payload, "wrinkle_competition_cross_rect")
            item["quality_flag"] = _quality_flag("zhe_zhou", box, circle, gray.shape, pixels_per_mm)
            out_put["result"] = [item]
            out_put["len"] = 1
            out_put["refine_status"] = "refined_wrinkle_cross"
            out_put["decision_source"] = "wrinkle_cross"
            return out_put
    if len(early_dark) >= 2 and not wrinkle_scene:
        out_put = dict(out_put)
        out_put["result"] = early_dark
        out_put["len"] = len(early_dark)
        out_put["refine_status"] = "refined_dark_components_no_wrinkle"
        out_put["decision_source"] = "dark_component"
        return out_put
    for item in result:
        name = item.get("class_name", "")
        if pred_time is None and item.get("prediction_time") is not None:
            pred_time = item.get("prediction_time")
        if name == "zang_wu":
            box = _bbox(item)
            if box:
                clipped = _clip_box(box, gray.shape)
                if clipped and _inside_ratio(clipped, circle) > 0.25:
                    z_boxes.append(clipped)
                    z_scores.append(_round_score(item.get("score", 0.75)))
            continue
        if name == "zhe_zhou":
            if _valid_wrinkle(gray, item, circle):
                item = dict(item)
                item["refined"] = True
                item["refine_mode"] = "refined_line"
                normal.append(item)
            continue
        normal.append(item)

    base_score = max(z_scores) if z_scores else 0.78
    refined_zang = _detect_dark_components(gray, circle, z_boxes, base_score, pred_time, pixels_per_mm)
    if refined_zang and _dark_components_are_confident_dirty(refined_zang, pixels_per_mm):
        normal.extend(refined_zang)
    else:
        # Conservative fallback: keep only model boxes mostly inside the sheet.
        for box, score in zip(z_boxes, z_scores):
            if _inside_ratio(box, circle) >= 0.88 and _box_center_inside(box, circle, 0.86):
                normal.append(_make_item("zang_wu", box, score, pred_time, "model_fallback"))

    if not normal:
        normal = [{"class_name": "zheng_chang", "score": 1.0, "loc": [0, 0, 1, 1], "refined": True, "refine_mode": "all_filtered"}]
    out_put = dict(out_put)
    out_put["result"] = normal
    out_put["len"] = len(normal)
    out_put["refine_status"] = "refined_sheet_components"
    return out_put
