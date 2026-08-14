import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from pathlib import Path
import threading
import queue
import json
from datetime import datetime
import pandas as pd
import shutil

from collections import Counter

from core.detect import find_plate_crop, load_plate_model
from core.decode import read_digit_boxes, select_digit_string, load_digit_model
from core.sequence import correct_records

class PlateOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plate OCR & Rename")
        self.root.geometry("800x600")

        self.input_folder = tk.StringVar()
        self.digit_count = tk.StringVar()  # опционально: 3 или 4, пусто = без нормализации длины
        plate_model_path = 'models/plate_detector.onnx'
        if not Path(plate_model_path).exists():
            raise FileNotFoundError(
                f"Не найдена модель детекции таблички: {plate_model_path}. "
                "Запусти download_models.py."
            )
        digit_model_path = 'models/digit_detector.onnx'
        if not Path(digit_model_path).exists():
            raise FileNotFoundError(
                f"Не найдена модель распознавания цифр: {digit_model_path}. "
                "Запусти download_models.py."
            )
        self.plate_model = load_plate_model(plate_model_path)
        self.digit_model = load_digit_model(digit_model_path)
        self.processing = False
        self.cancel_flag = threading.Event()
        self.log = None  # init log before use

        # process() крутится в отдельном потоке, а Tk не потокобезопасен -
        # никакие обращения к виджетам оттуда напрямую. Воркер только
        # кладёт сообщения в очередь, единственное место, которое реально
        # трогает виджеты, - _poll_log_queue на главном потоке (по таймеру
        # root.after).
        self.log_queue = queue.Queue()

        self.setup_ui()
        self._poll_log_queue()
        self.log_msg(f"✅ Loaded models: {plate_model_path}, {digit_model_path}")
    
    def setup_ui(self):
        tk.Label(self.root, text="Input folder:").pack(pady=5)
        entry = tk.Entry(self.root, textvariable=self.input_folder, width=80)
        entry.pack(pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_folder).pack(pady=5)

        tk.Label(self.root, text="Digit count on this session's plates (3 or 4, optional):").pack(pady=(10, 0))
        tk.Entry(self.root, textvariable=self.digit_count, width=6).pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Load", command=self.load_images).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Process", command=self.process_thread).pack(side=tk.LEFT, padx=5)
        self.cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.cancel_process, state='disabled')
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Export", command=self.export_renamed).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Undo last run", command=self.undo_last_run).pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=10)
        
        self.log = scrolledtext.ScrolledText(self.root, height=20)
        self.log.pack(fill='both', expand=True, padx=20, pady=10)
    
    def log_msg(self, msg):
        # Может звать и главный, и воркер-поток - кладём в очередь и всё,
        # никаких обращений к виджетам здесь.
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__DONE__":
                    self.progress.stop()
                    self.processing = False
                    self.cancel_btn.config(state='disabled')
                elif self.log:
                    self.log.insert(tk.END, item + '\n')
                    self.log.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder.set(folder)
    
    def load_images(self):
        self.log_msg("Loaded: " + self.input_folder.get())
    
    def process_thread(self):
        if self.processing:
            return
        self.processing = True
        self.cancel_flag.clear()
        self.cancel_btn.config(state='normal')
        self.progress.start()
        thread = threading.Thread(target=self.process)
        thread.daemon = True
        thread.start()
    
    def cancel_process(self):
        self.cancel_flag.set()
        self.log_msg("⏹️ Cancelling...")
    
    def process(self):
        try:
            input_path = Path(self.input_folder.get())
            # *.jpg и *.JPG совпадают на регистронезависимой ФС Windows -
            # убираем дубли по разрешённому пути.
            images = sorted({p.resolve() for p in (*input_path.glob('*.jpg'), *input_path.glob('*.JPG'))})

            self.log_msg(f"Found {len(images)} images")

            digit_count_str = self.digit_count.get().strip()
            manual_length = int(digit_count_str) if digit_count_str.isdigit() else None

            # Кроп режется из памяти (core.detect), поэтому связь
            # "результат -> исходное фото" не может потеряться - идём по
            # тому же списку images, что и печатаем в лог.
            # Цифры детектируем один раз за проход и запоминаем боксы -
            # финальную сборку кода (с учётом длины) делаем вторым,
            # дешёвым проходом, когда длина уже известна (вручную или
            # автоматически).
            detections = []
            for i, orig_img in enumerate(images):
                if self.cancel_flag.is_set():
                    self.log_msg("Cancelled by user")
                    break

                self.log_msg(f"Processing {i + 1}/{len(images)}: {orig_img.name}")

                crop, box_conf = find_plate_crop(orig_img, self.plate_model)
                if crop is None:
                    self.log_msg(f"  ⚠️ Табличка не найдена: {orig_img.name}")
                    continue

                digit_boxes = read_digit_boxes(crop, self.digit_model)
                detections.append((orig_img, digit_boxes))

            if manual_length:
                expected_length = manual_length
            else:
                # Автоопределение: длина без нормализации почти всегда
                # одинакова в пределах одной сессии (одна площадка) -
                # берём самую частую среди правдоподобных (3-4 цифры).
                geo_lengths = (len(select_digit_string(boxes)) for _, boxes in detections)
                lengths = Counter(n for n in geo_lengths if 3 <= n <= 4)
                expected_length = lengths.most_common(1)[0][0] if lengths else None
                if expected_length:
                    self.log_msg(f"Digit count не задан - определил автоматически: {expected_length}")

            ocr_results = []
            for orig_img, digit_boxes in detections:
                code = select_digit_string(digit_boxes, expected_length=expected_length)
                if not code:
                    self.log_msg(f"  ⚠️ Цифры не распознаны: {orig_img.name}")
                    continue
                ocr_results.append({'image': orig_img.name, 'path': orig_img, 'code': code})

            if expected_length:
                self.log_msg("Уточняю номера по порядку съёмки (EXIF)...")
                ocr_results = correct_records(ocr_results, expected_length)
            else:
                self.log_msg("Digit count не задан - уточнение по EXIF-последовательности пропущено")

            df = pd.DataFrame(ocr_results)[['image', 'code']] if ocr_results else pd.DataFrame(columns=['image', 'code'])
            df.to_csv('gui_plate_codes.csv', index=False)
            self.log_msg(f"✅ Распознано: {len(df)} из {len(images)} → gui_plate_codes.csv")

            # Rename (копия, оригиналы не трогаем)
            renamed_dir = input_path / 'renamed'
            renamed_dir.mkdir(exist_ok=True)

            manifest_entries = []
            for _, row in df.iterrows():
                if self.cancel_flag.is_set():
                    break
                orig_img = input_path / row['image']
                if orig_img.exists():
                    new_name = f"{row['code']}_{orig_img.name}"
                    shutil.copy2(orig_img, renamed_dir / new_name)
                    self.log_msg(f"✅ {orig_img.name} → {new_name}")
                    manifest_entries.append({"original": row['image'], "renamed": new_name})

            manifest = {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "entries": manifest_entries,
            }
            (renamed_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            self.log_msg(f"✅ Renamed {len(manifest_entries)} images → {renamed_dir}")
        except Exception as e:
            self.log_msg(f"❌ Error: {e}")
        finally:
            self.log_queue.put("__DONE__")
    
    def export_renamed(self):
        out_folder = filedialog.askdirectory(title="Export renamed")
        if out_folder:
            renamed_dir = Path(self.input_folder.get()) / 'renamed'
            if renamed_dir.exists() and len(list(renamed_dir.glob('*'))) > 0:
                shutil.copytree(renamed_dir, out_folder, dirs_exist_ok=True)
                self.log_msg(f"✅ Exported {len(list(renamed_dir.glob('*')))} files")
            else:
                messagebox.showerror("Error", "No renamed files!")

    def undo_last_run(self):
        """Удаляет файлы последнего прогона из renamed/ по manifest.json.
        Оригиналы не трогаются - они туда и не копировались."""
        renamed_dir = Path(self.input_folder.get()) / 'renamed'
        manifest_path = renamed_dir / "manifest.json"
        if not manifest_path.exists():
            self.log_msg("Отменять нечего - manifest.json не найден")
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        removed = 0
        for entry in manifest.get("entries", []):
            renamed_file = renamed_dir / entry["renamed"]
            if renamed_file.exists():
                renamed_file.unlink()
                removed += 1
        manifest_path.unlink()
        self.log_msg(f"↩️ Undo: удалено {removed} файлов из {renamed_dir} (прогон от {manifest.get('run_at')})")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateOCRApp(root)
    root.mainloop()

