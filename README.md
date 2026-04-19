[README.md](https://github.com/user-attachments/files/26870139/README.md)
# Photo Renamer - Автоматическое переименование фотографий по номерам на табличках

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)

## 🎯 Описание

Приложение для автоматического переименования фотографий по номерам на табличках. 

**Процесс:**
1. Детекция таблички с номером (YOLOv8)
2. Извлечение текста номера (EasyOCR) 
3. Переименование файла: `MDZ087_1.JPG` → `MDZ087_1.jpg`

Поддерживает GUI и CLI режимы.

## 🚀 Быстрый старт

```bash
git clone https://github.com/yourusername/sefer-photo-renamer.git
cd sefer-photo-renamer
pip install -r requirements.txt

# Скачать модели (ссылки в разделе Models)
python download_models.py  # digit_detector.pt
# Запуск GUI
python gui.py

# CLI: переименовать папку photos/
python app.py --source photos/ --dry-run
python app.py --source photos/
```

## 📦 Установка

```bash
pip install ultralytics easyocr opencv-python pandas pillow tkinter
```

### Модели (обязательно)

1. **plate_detector.pt** - YOLOv8 для детекции табличек [скачать](models/plate_detector.pt)
2. **digit_detector.pt** - Детектор цифр [скачать](models/digit_detector.pt)

Поместите в папку `models/`.

## 💻 Использование

### GUI режим
```
python gui.py
```
- Выберите папку с фото
- Настройте параметры (conf, OCR)
- Dry run / Rename

### CLI
```bash
python app.py --help

# Основные параметры
python app.py --source test/ --dry-run           # Показать изменения
python app.py --source test/                     # Переименовать
python app.py --source test/ --conf 0.5 --save-txt # Сохранить детекции
```

## 📊 Результаты

| Epoch | mAP50 | mAP50-95 | Model |
|-------|-------|----------|-------|
| 50    | 0.85  | 0.42     | [best.pt](models/plate_detector.pt) |

**Точность:** 85%+ на валидации (1 табличка/фото).

## 🛠 Структура проекта

```
.
├── app.py                 # CLI основной
├── gui.py                 # GUI интерфейс
├── models/                # Модели ML
│   ├── plate_detector.pt
│   └── digit_detector.pt
├── config/
│   └── data.yaml          # YOLO конфиг
├── requirements.txt       # Зависимости
├── README.md              # Документация
└── tests/                 # Тестовые данные
```

## 🔧 Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--conf` | 0.25 | Confidence порог YOLO |
| `--iou`  | 0.7  | NMS IoU threshold |
| `--dry-run` | - | Показать изменения БЕЗ сохранения |
| `--save-txt` | - | Сохранить YOLO labels |

## 📈 Метрики обучения

```
mAP50: 0.85
mAP50-95: 0.42 (50 эпох)
```

Графики обучения: [runs/detect/train/](runs/detect/train/)

## 🤝 Contributing

1. Fork репозитория
2. Создайте issue/feature request
3. Pull Request в `main`

## 📄 Лицензия

MIT License - смотрите [LICENSE](LICENSE)

## 🙏 Благодарности

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [LabelImg](https://github.com/HumanSignal/labelImg)

---

⭐ **Если помогло - поставьте звезду!**

