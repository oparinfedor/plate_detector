"""Чтение цифр с кропа таблички через digit_detector.onnx и сборка
итоговой строки номера.

Работает через onnxruntime напрямую, без ultralytics/torch в рантайме.
digit_detector - архитектура YOLO26 с "end-to-end" головой: экспорт в
ONNX уже включает NMS, выход - фиксированные (300, 6) строк
[x1, y1, x2, y2, conf, cls], лишний NMS вручную не нужен - только отсечь
по порогу уверенности. Класс-индексы - буквально цифры '0'..'9' (сверено
через model.names при экспорте .pt -> .onnx, onnxruntime такие имена не
хранит, поэтому маппинг зашит явно).

На кропе модель находит не только цифры номера, но и цифры
сантиметровой линейки под ним — их отсеивает select_digit_string
геометрически (по высоте и общей базовой линии), а не по смыслу.
"""
from core.onnx_backend import load_session, preprocess, unletterbox_xyxy

HEIGHT_RATIO_THRESHOLD = 0.65   # цифра меньше этой доли от макс. высоты - шум (линейка)
BASELINE_TOLERANCE_RATIO = 0.5  # допуск отклонения от базовой линии, в долях макс. высоты
OVERLAP_IOU_THRESHOLD = 0.3     # выше этого - считаем дублем одной и той же цифры
DIGIT_CONF_THRESHOLD = 0.25

DIGIT_CLASS_NAMES = [str(i) for i in range(10)]


def load_digit_model(path="models/digit_detector.onnx"):
    return load_session(path)


def read_digit_boxes(crop, session, conf=DIGIT_CONF_THRESHOLD):
    """Возвращает список {char, conf, x1, y1, x2, y2, w, h} для всех
    найденных на кропе "цифр" (без геометрической фильтрации)."""
    if crop is None or crop.size == 0:
        return []

    inp, scale, padx, pady = preprocess(crop)
    out = session.run(None, {"images": inp})[0][0]  # (300, 6)

    boxes = []
    for x1, y1, x2, y2, score, cls in out:
        if score < conf:
            continue
        cls_id = int(cls)
        char = DIGIT_CLASS_NAMES[cls_id] if 0 <= cls_id < len(DIGIT_CLASS_NAMES) else "?"
        x1, y1, x2, y2 = unletterbox_xyxy(x1, y1, x2, y2, scale, padx, pady)
        boxes.append({
            "char": char,
            "conf": float(score),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": x2 - x1, "h": y2 - y1,
        })
    return boxes


def select_digit_string(boxes, expected_length=None):
    """Геометрический отбор строки цифр номера из "сырых" боксов.

    1. Отбросить боксы мельче HEIGHT_RATIO_THRESHOLD от самой высокой
       (цифры линейки заметно мельче цифр номера).
    2. Отбросить боксы, чей низ заметно не совпадает с общей базовой
       линией оставшихся (линейка часто не на одном уровне с номером).
     3. Убрать дубли по горизонтальному перекрытию (одна и та же цифра
       иногда даёт две рамки) — оставляем более уверенную.
    4. Если задан expected_length и боксов больше - оставить
       expected_length самых крупных (шум обычно мельче настоящих
       цифр), потом досортировать по X. Если боксов меньше - дополнить
       результат нулями слева (ведущие нули часто не печатают на
       табличке отдельно, но подразумевают в номере).
    5. Отсортировать оставшееся слева направо и склеить в строку.
    """
    if not boxes:
        return ""

    max_h = max(b["h"] for b in boxes)
    filtered = [b for b in boxes if b["h"] >= HEIGHT_RATIO_THRESHOLD * max_h]
    if not filtered:
        return ""

    bottoms = sorted(b["y2"] for b in filtered)
    median_bottom = bottoms[len(bottoms) // 2]
    tolerance = BASELINE_TOLERANCE_RATIO * max_h
    filtered = [b for b in filtered if abs(b["y2"] - median_bottom) <= tolerance]
    if not filtered:
        return ""

    filtered = _dedupe_overlapping(filtered)

    if expected_length is not None and len(filtered) > expected_length:
        filtered = sorted(filtered, key=lambda b: b["h"], reverse=True)[:expected_length]

    filtered.sort(key=lambda b: b["x1"])
    digit_string = "".join(b["char"] for b in filtered)

    if expected_length is not None and len(digit_string) < expected_length:
        digit_string = digit_string.zfill(expected_length)

    return digit_string


def _dedupe_overlapping(boxes):
    boxes_by_conf = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in boxes_by_conf:
        if not any(_horizontal_iou(b, k) > OVERLAP_IOU_THRESHOLD for k in kept):
            kept.append(b)
    return kept


def _horizontal_iou(a, b):
    left = max(a["x1"], b["x1"])
    right = min(a["x2"], b["x2"])
    inter = max(0.0, right - left)
    if inter == 0:
        return 0.0
    union = (a["x2"] - a["x1"]) + (b["x2"] - b["x1"]) - inter
    return inter / union if union > 0 else 0.0
