"""Честный train/val split для датасета детектора таблички.

Раньше config/data.yaml и train/data_new.yaml указывали val на ту же
папку, что и train (val==train) - mAP отчёта в таком случае измеряет,
насколько модель запомнила обучающую выборку, а не как она обобщается.

Разбивать по отдельным фотографиям нельзя: несколько кадров одной и той
же таблички (train/images/MDZ086_1_JPG.rf....JPG,
MDZ086_2_JPG.rf....JPG, ...) почти дубликаты - случайный сплит по файлам
пропустит один кадр таблички в train, другой в val, и оценка снова
соврёт. Поэтому группируем по идентичности таблички (площадка+номер,
без суффикса кадра/хэша Roboflow) и делим 80/20 по группам.

Запуск: python prepare_split.py
Пишет: train/split_train.txt, train/split_val.txt, train/data_honest.yaml
"""
import random
import re
from pathlib import Path

IMAGES_DIR = Path("train/images")
OUT_DIR = Path("train")

# MDZ086_1_JPG.rf.<hash>.JPG -> group id "MDZ086"; QBA1096a_2_JPG.rf...
# -> "QBA1096A" (буква после номера тоже часть идентичности таблички)
PLATE_ID_RE = re.compile(r"^([A-Za-z]+\d+[a-zA-Z]?)(?:\s?_\d+)?_JPG\.rf\.", re.IGNORECASE)

VAL_FRACTION = 0.2
SEED = 0


def group_images():
    groups = {}
    unmatched = []
    for p in sorted(IMAGES_DIR.glob("*.JPG")):
        m = PLATE_ID_RE.match(p.name)
        if not m:
            unmatched.append(p)
            continue
        plate_id = m.group(1).upper()
        groups.setdefault(plate_id, []).append(p)
    if unmatched:
        raise RuntimeError(
            f"Не удалось разобрать идентичность таблички для {len(unmatched)} "
            f"файлов (первые несколько: {[p.name for p in unmatched[:5]]}) - "
            "проверь PLATE_ID_RE, иначе эти фото не попадут ни в train, ни в val."
        )
    return groups


def split_groups(groups):
    plate_ids = sorted(groups.keys())
    rng = random.Random(SEED)
    rng.shuffle(plate_ids)

    n_val_groups = max(1, round(len(plate_ids) * VAL_FRACTION))
    val_ids = set(plate_ids[:n_val_groups])
    train_ids = set(plate_ids[n_val_groups:])

    train_images = [p for pid in train_ids for p in groups[pid]]
    val_images = [p for pid in val_ids for p in groups[pid]]
    return train_images, val_images, train_ids, val_ids


def write_split(train_images, val_images):
    # Абсолютные пути - не полагаемся на то, как именно ultralytics
    # резолвит относительные пути внутри txt-списка (это недокументированная
    # тонкость), надёжнее один раз прописать однозначно.
    train_txt = OUT_DIR / "split_train.txt"
    val_txt = OUT_DIR / "split_val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_images) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_images) + "\n", encoding="utf-8")

    # ultralytics резолвит path: относительно рабочей директории запуска
    # скрипта (не относительно самого yaml-файла) - скрипты этого проекта
    # всегда запускаются из корня, поэтому path: train.
    data_yaml = OUT_DIR / "data_honest.yaml"
    data_yaml.write_text(
        "path: train\n"
        "train: split_train.txt\n"
        "val: split_val.txt\n"
        "\n"
        "nc: 1\n"
        "names: ['plate']\n",
        encoding="utf-8",
    )
    return train_txt, val_txt, data_yaml


if __name__ == "__main__":
    groups = group_images()
    train_images, val_images, train_ids, val_ids = split_groups(groups)

    assert train_ids.isdisjoint(val_ids), "утечка: одна и та же табличка в train и val"

    train_txt, val_txt, data_yaml = write_split(train_images, val_images)

    print(f"Табличек всего: {len(groups)} (train: {len(train_ids)}, val: {len(val_ids)})")
    print(f"Изображений: train={len(train_images)}, val={len(val_images)}")
    print(f"Написано: {train_txt}, {val_txt}, {data_yaml}")
