"""Находит табличку с номером на фото и возвращает один лучший кроп.

Кроп режется прямо из массива numpy, который YOLO и так держит в
памяти (result.orig_img) — на диск ничего не пишется и не читается
обратно, поэтому связь "кроп -> исходное фото" не может потеряться
(в отличие от старой схемы через save_crop + угадывание имени файла).
"""


def find_plate_crop(image_path, plate_model, conf=0.2):
    """Возвращает (crop, box_conf) для лучшего бокса класса 'plate',
    либо (None, None), если табличка не найдена."""
    results = plate_model.predict(
        source=str(image_path), conf=conf, classes=[0], verbose=False
    )
    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[best_idx].tolist())
    box_conf = float(boxes.conf[best_idx])

    crop = result.orig_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, box_conf
