"""Находит табличку с номером на фото и возвращает один лучший кроп.

Работает через onnxruntime напрямую, без ultralytics/torch в рантайме.
plate_detector.onnx - обычная YOLO-модель с "сырым" выходом (1,5,8400):
на каждый из 8400 якорей - (cx,cy,w,h,score) для единственного класса
'plate'. Полноценный NMS не нужен - нам всегда нужен только один лучший
по уверенности бокс, поэтому декодируем через argmax по score.
"""
import cv2

from core.onnx_backend import load_session, preprocess, unletterbox_xyxy

CONF_THRESHOLD = 0.2


def load_plate_model(path="models/plate_detector.onnx"):
    return load_session(path)


def find_plate_crop(image_path, session, conf=CONF_THRESHOLD):
    """Возвращает (crop, box_conf) для лучшего бокса класса 'plate',
    либо (None, None), если табличка не найдена."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    inp, scale, padx, pady = preprocess(img)
    out = session.run(None, {"images": inp})[0]  # (1, 5, 8400)
    out = out[0].T  # (8400, 5): cx, cy, w, h, score

    best_idx = int(out[:, 4].argmax())
    cx, cy, w, h, score = out[best_idx]
    if score < conf:
        return None, None

    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    x1, y1, x2, y2 = unletterbox_xyxy(x1, y1, x2, y2, scale, padx, pady)

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, float(score)
