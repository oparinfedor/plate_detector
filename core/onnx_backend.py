"""Общая инфраструктура инференса через onnxruntime - без зависимости от
ultralytics/torch в рантайме приложения (аудит: экспорт в ONNX убирает
~2.5 ГБ зависимостей torch из офлайн-поставки и ускоряет инференс на CPU).

Letterbox-препроцессинг воспроизводит поведение ultralytics по умолчанию
для одиночного изображения (`predict(..., rect=True)`) - паддинг только до
ближайшего кратного stride (32), а не до полного квадрата INPUT_SIZE.
Это принципиально: модели экспортированы с `dynamic=True`, и на сильно
неквадратных кропах (табличка на кропе шире, чем выше) паддинг в полный
квадрат размывает уверенность модели на пограничных детекциях - численно
проверено на реальном фото: одна и та же цифра при паддинге в квадрат
давала conf=0.016, а при прямоугольном паддинге (как делает ultralytics
по умолчанию) - conf=0.695, то есть то же самое, что выдаёт torch.
"""
import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = 640
PAD_COLOR = 114
STRIDE = 32


def load_session(onnx_path):
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def letterbox(img, new_size=INPUT_SIZE, stride=STRIDE):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Паддинг только до ближайшего кратного stride (минимальный
    # прямоугольник), а не до полного new_size x new_size квадрата -
    # см. пояснение в докстринге модуля.
    dh, dw = (-nh) % stride, (-nw) % stride
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(PAD_COLOR, PAD_COLOR, PAD_COLOR),
    )
    return canvas, scale, left, top


def preprocess(img):
    canvas, scale, padx, pady = letterbox(img)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[None, ...], scale, padx, pady


def unletterbox_xyxy(x1, y1, x2, y2, scale, padx, pady):
    return (x1 - padx) / scale, (y1 - pady) / scale, (x2 - padx) / scale, (y2 - pady) / scale
