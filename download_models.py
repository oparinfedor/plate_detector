#!/usr/bin/env python3
"""Download digit_detector.pt (118MB)"""
import os
from pathlib import Path

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
model_path = models_dir / 'digit_detector.pt'

GDRIVE_ID = '1yfe9spwAbQiy-xeQ2w_GafGQmMtLKugq'

print('🔍 digit_detector.pt...')
if model_path.exists():
    print('✅ Ready!')
else:
    print('❌ Download manually:')
    print(f'https://drive.google.com/uc?export=download&id={GDRIVE_ID}')
    print('→ models/digit_detector.pt')

print('🚀 python plate_ocr_gui_final.py')
