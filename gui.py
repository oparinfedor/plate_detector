import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import cv2
from pathlib import Path
import threading
from ultralytics import YOLO
import easyocr
import pandas as pd
import shutil
import re

class PlateOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plate OCR & Rename")
        self.root.geometry("800x600")
        
        self.input_folder = tk.StringVar()
        model_path = 'runs/detect/train/weights/best.pt' if Path('runs/detect/train/weights/best.pt').exists() else 'yolo26n.pt'
        self.model = YOLO(model_path)
        self.ocr_reader = easyocr.Reader(['en'])
        self.processing = False
        self.cancel_flag = threading.Event()
        self.log = None  # init log before use
        
        self.setup_ui()
        self.log_msg(f"✅ Loaded model: {model_path}")
    
    def setup_ui(self):
        tk.Label(self.root, text="Input folder:").pack(pady=5)
        entry = tk.Entry(self.root, textvariable=self.input_folder, width=80)
        entry.pack(pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_folder).pack(pady=5)
        
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
            images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.JPG'))
            
            self.log_msg(f"Found {len(images)} images")
            
            import tempfile
            import os
            
            # Create temp dir for crops
            temp_dir = Path(tempfile.mkdtemp(prefix='plate_crops_'))
            self.log_msg(f"Temp crops dir: {temp_dir}")
            
            # YOLO predict to temp, only plate class (classes=0)
            results = self.model.predict(images, conf=0.1, save_crop=True, project=str(temp_dir), name='predict', classes=0)
            
            # Map crops to original images
            crop_to_orig = {}
            for r in results:
                orig_name = Path(r.path).name
                for crop_path in r.boxes.data:  # Actually from crop filenames or r.save_dir
                    # YOLO crops are named orig_name_cropped.jpg in crops/plate/
                    crop_name = orig_name.replace('.jpg', '_cropped.jpg')  # Adjust pattern
                    crop_to_orig[crop_name] = orig_name
            
            # Find plate crops
            plate_crops_dir = temp_dir / 'predict' / 'crops' / 'plate'
            crop_files = list(plate_crops_dir.glob('*.jpg'))
            self.log_msg(f"Found {len(crop_files)} plate crops")
            
            if len(crop_files) == 0:
                self.log_msg("❌ No plate crops generated!")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return
            
            # OCR & codes
            ocr_results = []
            for i, crop_file in enumerate(crop_files[:100]):
                if self.cancel_flag.is_set():
                    self.log_msg("Cancelled by user")
                    break
                
                orig_name = crop_to_orig.get(crop_file.name, 'unknown')
                self.log_msg(f"OCR {i+1}/{len(crop_files)}: {orig_name}")
                img = cv2.imread(str(crop_file))
                if img is None:
                    continue
                ocr_res = self.ocr_reader.readtext(img)
                numbers_str = ' '.join([t[1] for t in ocr_res if t[2] > 0.5])
                numbers = re.findall(r'\d+', numbers_str)
                code = ''.join(numbers[:4]).zfill(4)
                ocr_results.append({'image': crop_file.name, 'code': code})
            
            df = pd.DataFrame(ocr_results)
            df.to_csv('gui_plate_codes.csv', index=False)
            self.log_msg(f"✅ OCR: {len(df)} codes → gui_plate_codes.csv")
            
            # Rename
            renamed_dir = input_path / 'renamed'
            renamed_dir.mkdir(exist_ok=True)
            
            renamed_count = 0
            for _, row in df.iterrows():
                if self.cancel_flag.is_set():
                    break
                base_name = row['image'].split('_')[0]
                orig_img = next(input_path.glob(f"{base_name}*"), None)
                if orig_img and orig_img.exists():
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
            # Cleanup temp dir
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.log_msg(f"🧹 Cleaned temp dir: {temp_dir}")
    
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

