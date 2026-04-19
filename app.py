import pandas as pd
import shutil
from pathlib import Path

codes_csv = 'gui_plate_codes.csv'
input_dir = Path('test')
backup_dir = input_dir / 'backup'
new_dir = input_dir / 'renamed'

input_dir.mkdir(exist_ok=True)
backup_dir.mkdir(exist_ok=True)
new_dir.mkdir(exist_ok=True)

df = pd.read_csv(codes_csv)

print(f"Processing {len(df)} codes from {codes_csv}")

renamed_count = 0
for idx, row in df.iterrows():
    old_name = row['image']
    code = row['code']
    
    old_path = input_dir / old_name
    if not old_path.exists():
        print(f"❌ Skip: {old_name} not found")
        continue
    
    # Backup original
    backup_path = backup_dir / old_name
    shutil.copy2(old_path, backup_path)
    
    # New name
    new_name = f"{code}_{old_name}"
    new_path = new_dir / new_name
    
    # Rename (move)
    shutil.move(str(old_path), str(new_path))
    print(f"✅ {old_name} → {new_name}")
    renamed_count += 1

print(f"\n🎉 Renamed {renamed_count}/{len(df)} files!")
print(f"📁 Backup: {backup_dir}")
print(f"📁 Renamed: {new_dir}")
print("Done! 🎯")
