# Photo Renamer — автоматическое переименование фотографий по номерам на табличках

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-green.svg)](https://github.com/ultralytics/ultralytics)

## Описание

Инструмент для полевой обработки фотографий: переименовывает снимки по
номеру, который виден на табличке-нумераторе в кадре.

**Процесс:**
1. Детекция таблички в кадре (`models/plate_detector.onnx`, YOLO) — берётся
   один лучший по уверенности бокс, кроп режется прямо из памяти
2. Распознавание цифр на кропе специализированной моделью
   (`models/digit_detector.onnx`) — не общий OCR, а YOLO-детектор, обученный
   именно на такие цифры
3. Геометрический отбор найденных боксов (отсекает шум вроде делений
   сантиметровой линейки на нумераторе) и сборка номера слева направо
4. Если в папке много фото одной камерой подряд — необязательное
   уточнение номеров по порядку съёмки через EXIF (см. поле
   "Digit count" ниже)
5. Копия файла с новым именем `<номер>_<исходное_имя>` сохраняется в
   `<папка>/renamed/` — оригиналы никогда не трогаются

Инструмент только с графическим интерфейсом — отдельного CLI сейчас нет.

Рантайм работает через onnxruntime напрямую, без torch/ultralytics в
поставке (~2.5 ГБ зависимостей убрано из офлайн-сборки). Ultralytics
нужен только для обучения/переобучения моделей (`requirements-train.txt`),
не для самого приложения.

## Быстрый старт

```bash
git clone https://github.com/oparinfedor/plate_detector.git
cd plate_detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

python download_models.py      # digit_detector.onnx (215 МБ) - см. ниже
python gui.py
```

## Модели

- `models/plate_detector.onnx` — уже в репозитории. Обучен на честном
  train/val split (mAP50 = 0.995, mAP50-95 = 0.795 на реально невиданных
  фото — см. раздел «Точность»).
- `models/digit_detector.onnx` — 215 МБ, слишком тяжёлый для обычного git,
  качается отдельно: `python download_models.py` выведет прямую ссылку
  (Яндекс.Диск), файл нужно положить в `models/` вручную.

## Использование

```bash
python gui.py
```

- **Browse** — выбрать папку с фотографиями
- **Digit count** — сколько цифр на табличках в этой сессии (3 или 4).
  Можно оставить пустым — определится автоматически по большинству
  кадров в папке. Если поле заполнено, дополнительно включается
  уточнение по EXIF-последовательности съёмки (шаг 4 выше)
- **Process** — запускает распознавание; результаты копируются в
  `<папка>/renamed/` вместе с журналом `manifest.json`
- **Undo last run** — удаляет из `renamed/` файлы последнего прогона по
  `manifest.json` (оригиналы не затрагиваются в любом случае)
- **Export** — скопировать содержимое `renamed/` в другое место
- **Cancel** — прервать текущую обработку

## Сборка exe (PyInstaller)

Для офлайн-раздачи без установки Python:

```bash
pip install -r requirements.txt pyinstaller

pyinstaller --name PlateOCR --onedir --windowed --noconfirm ^
  --exclude-module torch --exclude-module torchvision --exclude-module torchaudio ^
  --exclude-module ultralytics --exclude-module matplotlib --exclude-module onnx ^
  --exclude-module sympy --exclude-module scipy gui.py
```

`--exclude-module` обязателен: без него PyInstaller статически находит
`import torch` внутри необязательного `onnxruntime.transformers.machine_info`
(не используется этим приложением) и на всякий случай тащит в сборку весь
torch/torchvision — это добавляет ~0.4 ГБ мёртвого веса и противоречит
всему смыслу перехода на onnxruntime. С исключениями сборка (`dist/PlateOCR/`)
занимает ~0.23 ГБ.

После сборки положить обе модели в `dist/PlateOCR/models/`
(`plate_detector.onnx` из репозитория, `digit_detector.onnx` — см.
раздел «Модели») и раздавать папку `dist/PlateOCR/` целиком.
`PlateOCR.spec` в репозитории фиксирует эту конфигурацию.

## Точность

Измерено скриптом `core/eval.py` на `test/` (1037 фото, эталон — номер
зашит в имя файла):

| Стадия                                  | Точность |
|------------------------------------------|---------:|
| Без отбора (сырой вывод детектора цифр)   |     0.8% |
| + геометрический отбор цифр               |    39.1% |
| + нормализация по длине номера            |    79.0% |
| + уточнение по EXIF-последовательности    | **85.1%** |

Детектор таблички: mAP50 = 0.995, mAP50-95 = 0.795 — честный
train/val split по идентичности таблички (не по отдельным кадрам, чтобы
несколько фото одной таблички не утекали одновременно в train и val),
см. `prepare_split.py`.

Прогнать самостоятельно: `python -m core.eval` (опция `--sample N` для
быстрой проверки на подвыборке).

## Структура проекта

```
core/
  onnx_backend.py            общая инфраструктура onnxruntime-инференса (letterbox, сессии)
  detect.py                  находит табличку, отдаёт один лучший кроп (в памяти)
  decode.py                  digit_detector + геометрический отбор + нормализация
  sequence.py                уточнение номеров по EXIF-последовательности
  eval.py                    точность на test/ (все стадии пайплайна)
gui.py                      основной GUI-инструмент
app.py                      вспомогательный скрипт: переименование по gui_plate_codes.csv
prepare_split.py            честный train/val split для переобучения детектора
retrain_plate_detector.py   переобучение детектора таблички + экспорт в onnx
download_models.py          скачивание digit_detector.onnx
config/data.yaml            конфиг честного split для YOLO-обучения
models/                     веса (plate_detector.onnx в репозитории, digit_detector.onnx - отдельно)
requirements.txt            рантайм (onnxruntime, opencv, pandas, pillow)
requirements-train.txt      + ultralytics/onnx, только для обучения/экспорта
TODO.md                     что ещё не сделано
```

## Дальнейшие планы

См. `TODO.md`.

## Лицензия

MIT — см. [LICENSE](LICENSE).

## Благодарности

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [LabelImg](https://github.com/HumanSignal/labelImg) — разметка датасета
