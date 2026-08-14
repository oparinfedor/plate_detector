import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from pathlib import Path
import sys
import threading
import queue
import json
from datetime import datetime
import pandas as pd
import shutil
from PIL import Image, ImageTk

from collections import Counter

from core.detect import find_plate_crop, load_plate_model
from core.decode import read_digit_boxes, select_digit_string, load_digit_model
from core.sequence import correct_records

# В собранном PyInstaller-приложении текущая рабочая директория - не
# обязательно папка с exe (например, при запуске ярлыком) - модели и
# другие файлы рядом с приложением ищем относительно самого exe/скрипта,
# а не CWD.
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

class PlateOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plate OCR & Rename")
        self.root.geometry("800x600")

        self.input_folder = tk.StringVar()
        self.digit_count = tk.StringVar()  # опционально: 3 или 4, пусто = без нормализации длины
        plate_model_path = BASE_DIR / 'models' / 'plate_detector.onnx'
        if not plate_model_path.exists():
            raise FileNotFoundError(
                f"Не найдена модель детекции таблички: {plate_model_path}. "
                "Запусти download_models.py."
            )
        digit_model_path = BASE_DIR / 'models' / 'digit_detector.onnx'
        if not digit_model_path.exists():
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
                elif isinstance(item, tuple) and item[0] == "__REVIEW__":
                    _, review_items, input_path = item
                    self.open_review_screen(review_items, input_path)
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

            # Экран ручной проверки решает, что реально переименовывать -
            # здесь только собираем список по всем images (включая те, где
            # табличка/цифры не нашлись - их код останется пустым, чтобы
            # можно было вписать вручную), в исходном порядке съёмки.
            code_by_name = {r['image']: r['code'] for r in ocr_results}
            review_items = [
                {'image': orig_img.name, 'path': orig_img, 'code': code_by_name.get(orig_img.name, '')}
                for orig_img in images
            ]
            recognized = sum(1 for item in review_items if item['code'])
            self.log_msg(f"✅ Распознано: {recognized} из {len(images)}. Открываю окно проверки...")
            self.log_queue.put(("__REVIEW__", review_items, input_path))
        except Exception as e:
            self.log_msg(f"❌ Error: {e}")
        finally:
            self.log_queue.put("__DONE__")

    def open_review_screen(self, review_items, input_path):
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Проверка результатов ({len(review_items)})")
        dialog.geometry("640x600")
        dialog.transient(self.root)
        dialog.grab_set()

        canvas = tk.Canvas(dialog, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient='vertical', command=canvas.yview)
        rows_frame = tk.Frame(canvas)
        rows_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=rows_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        def on_mousewheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        dialog.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side='top', fill='both', expand=True, padx=10, pady=(10, 0))
        scrollbar.pack(side='right', fill='y')

        tk.Label(rows_frame, text="Фото", width=14).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(rows_frame, text="Файл").grid(row=0, column=1, sticky='w', padx=5, pady=5)
        tk.Label(rows_frame, text="Номер", width=12).grid(row=0, column=2, padx=5, pady=5)

        thumb_size = (96, 72)
        photos = []  # ссылки на PhotoImage - иначе Tk соберёт их как мусор
        rows = []  # (item, StringVar) для сборки финальных кодов при подтверждении
        for i, item in enumerate(review_items, start=1):
            try:
                img = Image.open(item['path'])
                img.thumbnail(thumb_size)
                photo = ImageTk.PhotoImage(img)
            except Exception:
                photo = None
            photos.append(photo)

            if photo:
                tk.Label(rows_frame, image=photo).grid(row=i, column=0, padx=5, pady=3)
            else:
                tk.Label(rows_frame, text="—", width=14).grid(row=i, column=0, padx=5, pady=3)

            tk.Label(rows_frame, text=item['image'], anchor='w').grid(row=i, column=1, sticky='w', padx=5)

            code_var = tk.StringVar(value=item['code'])
            tk.Entry(rows_frame, textvariable=code_var, width=12).grid(row=i, column=2, padx=5, pady=3)
            rows.append((item, code_var))

        self._review_photos = photos  # держим ссылки на весь срок жизни окна

        def cleanup():
            dialog.unbind_all("<MouseWheel>")

        def on_confirm():
            cleanup()
            dialog.destroy()
            self._finalize_review(rows, input_path)

        def on_cancel():
            cleanup()
            dialog.destroy()
            self.log_msg("↩️ Проверка отменена - ничего не переименовано")

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(side='bottom', fill='x', padx=10, pady=10)
        tk.Button(btn_frame, text="Переименовать", command=on_confirm).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Отмена", command=on_cancel).pack(side='left', padx=5)

    def _finalize_review(self, rows, input_path):
        final = [(item, var.get().strip()) for item, var in rows]

        df = pd.DataFrame(
            [{'image': item['image'], 'code': code} for item, code in final if code],
            columns=['image', 'code'],
        )
        codes_csv = input_path / 'gui_plate_codes.csv'
        df.to_csv(codes_csv, index=False)
        self.log_msg(f"✅ Подтверждено: {len(df)} из {len(final)} → {codes_csv}")

        renamed_dir = input_path / 'renamed'
        renamed_dir.mkdir(exist_ok=True)

        manifest_entries = []
        for item, code in final:
            if not code:
                continue
            orig_img = item['path']
            if orig_img.exists():
                new_name = f"{code}_{orig_img.name}"
                shutil.copy2(orig_img, renamed_dir / new_name)
                self.log_msg(f"✅ {orig_img.name} → {new_name}")
                manifest_entries.append({"original": item['image'], "renamed": new_name})

        manifest = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "entries": manifest_entries,
        }
        (renamed_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.log_msg(f"✅ Renamed {len(manifest_entries)} images → {renamed_dir}")
    
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

