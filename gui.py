import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from pathlib import Path
import threading
from ultralytics import YOLO
import pandas as pd
import shutil

from core.detect import find_plate_crop
from core.decode import read_digit_boxes, select_digit_string

class PlateOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plate OCR & Rename")
        self.root.geometry("800x600")

        self.input_folder = tk.StringVar()
        self.digit_count = tk.StringVar()  # опционально: 3 или 4, пусто = без нормализации длины
        plate_model_path = 'models/plate_detector.pt'
        if not Path(plate_model_path).exists():
            raise FileNotFoundError(
                f"Не найдена модель детекции таблички: {plate_model_path}. "
                "Запусти download_models.py."
            )
        digit_model_path = 'models/digit_detector.pt'
        if not Path(digit_model_path).exists():
            raise FileNotFoundError(
                f"Не найдена модель распознавания цифр: {digit_model_path}. "
                "Запусти download_models.py."
            )
        self.plate_model = YOLO(plate_model_path)
        self.digit_model = YOLO(digit_model_path)
        self.processing = False
        self.cancel_flag = threading.Event()
        self.log = None  # init log before use

        self.setup_ui()
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
        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=10)
        
        self.log = scrolledtext.ScrolledText(self.root, height=20)
        self.log.pack(fill='both', expand=True, padx=20, pady=10)
    
    def log_msg(self, msg):
        if self.log:
            self.log.insert(tk.END, msg + '\n')
            self.log.see(tk.END)
        self.root.update()
    
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
            expected_length = int(digit_count_str) if digit_count_str.isdigit() else None

            # Кроп режется из памяти (core.detect), поэтому связь
            # "результат -> исходное фото" не может потеряться - идём по
            # тому же списку images, что и печатаем в лог.
            ocr_results = []
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
                code = select_digit_string(digit_boxes, expected_length=expected_length)
                if not code:
                    self.log_msg(f"  ⚠️ Цифры не распознаны: {orig_img.name}")
                    continue

                ocr_results.append({'image': orig_img.name, 'code': code})

            df = pd.DataFrame(ocr_results)
            df.to_csv('gui_plate_codes.csv', index=False)
            self.log_msg(f"✅ Распознано: {len(df)} из {len(images)} → gui_plate_codes.csv")

            # Rename
            renamed_dir = input_path / 'renamed'
            renamed_dir.mkdir(exist_ok=True)

            renamed_count = 0
            for _, row in df.iterrows():
                if self.cancel_flag.is_set():
                    break
                orig_img = input_path / row['image']
                if orig_img.exists():
                    new_name = f"{row['code']}_{orig_img.name}"
                    shutil.copy2(orig_img, renamed_dir / new_name)
                    self.log_msg(f"✅ {orig_img.name} → {new_name}")
                    renamed_count += 1

            self.log_msg(f"✅ Renamed {renamed_count} images → {renamed_dir}")
        except Exception as e:
            self.log_msg(f"❌ Error: {e}")
        finally:
            self.progress.stop()
            self.processing = False
            self.cancel_btn.config(state='disabled')
    
    def export_renamed(self):
        out_folder = filedialog.askdirectory(title="Export renamed")
        if out_folder:
            renamed_dir = Path(self.input_folder.get()) / 'renamed'
            if renamed_dir.exists() and len(list(renamed_dir.glob('*'))) > 0:
                shutil.copytree(renamed_dir, out_folder, dirs_exist_ok=True)
                self.log_msg(f"✅ Exported {len(list(renamed_dir.glob('*')))} files")
            else:
                messagebox.showerror("Error", "No renamed files!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateOCRApp(root)
    root.mainloop()

