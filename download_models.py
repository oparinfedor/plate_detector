#!/usr/bin/env python3
"""Download digit_detector.onnx (215MB)"""
from pathlib import Path

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
model_path = models_dir / 'digit_detector.onnx'

YANDEX_DISK_URL = 'https://disk.yandex.ru/d/YDegP1z5shaKQw'

print('🔍 digit_detector.onnx...')
if model_path.exists():
    print('✅ Ready!')
else:
    print('❌ Download manually:')
    print(YANDEX_DISK_URL)
    print('→ models/digit_detector.onnx')

print('🚀 python gui.py')
