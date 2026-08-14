"""Численная точность пайплайна детекции+распознавания против имён
файлов в test/ (эталон зашит в имя: TUL867, MDZ130_1, QBA1156_11...).

Запуск:
    python -m core.eval [--sample N] [--test-dir test]

Печатает точность на трёх стадиях (без отбора / с геометрическим
отбором / с отбором + нормализацией длины) — чтобы можно было свериться
с лестницей из аудита (0% -> 31% -> 51% -> 54%) и убедиться, что фикс
работает, а не просто что-то совпало случайно. Расхождения на финальной
стадии сохраняются в eval_mismatches.csv.
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from core.detect import find_plate_crop, load_plate_model
from core.decode import read_digit_boxes, select_digit_string, load_digit_model
from core.sequence import read_camera_id, read_capture_time, viterbi_correct

# Имена в test/ не единообразны, встречались:
#   003bb334-TUL867.JPG        (с 8-символьным hex-префиксом)
#   MDZ087.JPG                 (без префикса)
#   MDZ087 _3.JPG               (пробел перед _N)
#   QBA1156_11.JPG
GROUND_TRUTH_RE = re.compile(
    r"^(?:[0-9a-fA-F]{8}-)?([A-Za-z]{2,4})(\d{3,4})(?:\s?_\d+)?\.(?:jpe?g)$",
    re.IGNORECASE,
)


def parse_ground_truth(filename):
    m = GROUND_TRUTH_RE.match(filename)
    return (m.group(1).upper(), m.group(2)) if m else (None, None)


def raw_digit_string(boxes):
    """Без какой-либо фильтрации - baseline "digit_detector без отбора"."""
    ordered = sorted(boxes, key=lambda b: b["x1"])
    return "".join(b["char"] for b in ordered)


def run_eval(test_dir, sample=None, plate_model_path="models/plate_detector.onnx",
             digit_model_path="models/digit_detector.onnx"):
    test_dir = Path(test_dir)
    # На Windows файловая система регистронезависима, поэтому *.jpg и
    # *.JPG находят одни и те же файлы дважды - убираем дубли по
    # разрешённому пути, а не полагаемся на регистр расширения.
    images = sorted({p.resolve() for p in (*test_dir.glob("*.jpg"), *test_dir.glob("*.JPG"))})

    parsed = [(p, *parse_ground_truth(p.name)) for p in images]
    unparsable = [p for p, prefix, gt in parsed if gt is None]
    cases = [(p, prefix, gt) for p, prefix, gt in parsed if gt is not None]

    if sample:
        random.seed(0)
        cases = random.sample(cases, min(sample, len(cases)))

    print(f"Всего файлов: {len(images)}, эталон разобран: {len(cases)}, "
          f"не разобрано (пропущено): {len(unparsable)}")

    plate_model = load_plate_model(plate_model_path)
    digit_model = load_digit_model(digit_model_path)

    no_crop = 0
    raw_correct = 0
    geo_correct = 0
    norm_correct = 0
    ceiling_hits = 0
    mismatches = []
    # для стадии Витерби: группируем по (камера, площадка) - это ближе к
    # тому, как реально используется core.sequence на одной папке за
    # раз, чем группировка по всему test/ разом (там намешаны сессии)
    by_group = defaultdict(list)

    for path, prefix, gt in cases:
        crop, _ = find_plate_crop(path, plate_model)
        if crop is None:
            no_crop += 1
            mismatches.append((path.name, gt, "", "", "нет кропа"))
            by_group_key = (read_camera_id(path), prefix)
            by_group[by_group_key].append((read_capture_time(path), path, gt, ""))
            continue

        boxes = read_digit_boxes(crop, digit_model)

        raw = raw_digit_string(boxes)
        geo = select_digit_string(boxes)
        norm = select_digit_string(boxes, expected_length=len(gt))

        if raw == gt:
            raw_correct += 1
        if geo == gt:
            geo_correct += 1
        if norm == gt:
            norm_correct += 1
        else:
            mismatches.append((path.name, gt, geo, norm, ""))

        if gt in raw:
            # "потолок" = верный номер присутствует в сырой,
            # нефильтрованной строке цифр. norm может формально
            # совпасть с эталоном и без этого (например, если ведущий
            # ноль не был найден вообще и появился только за счёт
            # zfill) - это отдельный, обоснованный источник точности,
            # а не то же самое, что "цифры физически были в выдаче".
            ceiling_hits += 1

        group_key = (read_camera_id(path), prefix)
        by_group[group_key].append((read_capture_time(path), path, gt, norm))

    n = len(cases)
    if n == 0:
        print("Нет размеченных файлов для оценки.")
        return

    # Стадия Витерби: внутри каждой (камера, площадка) группы, отсортированной
    # по времени съёмки, прогоняем скорректированную последовательность.
    viterbi_correct_count = 0
    for (camera, prefix), items in by_group.items():
        items = [it for it in items if it[0] is not None]
        if len(items) < 3 or not camera:
            for _, _, gt, norm in items:
                if norm == gt:
                    viterbi_correct_count += 1
            continue
        items.sort(key=lambda it: it[0])
        widths = Counter(len(gt) for _, _, gt, _ in items)
        expected_length = widths.most_common(1)[0][0]
        observations = [norm for _, _, _, norm in items]
        fixed = viterbi_correct(observations, expected_length)
        for (_, _, gt, _), val in zip(items, fixed):
            predicted = str(val).zfill(expected_length) if val is not None else ""
            if predicted == gt:
                viterbi_correct_count += 1

    print(f"\nТочность (без отбора):                  {raw_correct}/{n} = {100 * raw_correct / n:.1f}%")
    print(f"Точность (+ геометрический отбор):       {geo_correct}/{n} = {100 * geo_correct / n:.1f}%")
    print(f"Точность (+ нормализация длины):         {norm_correct}/{n} = {100 * norm_correct / n:.1f}%")
    print(f"Точность (+ Витерби по камере+времени):  {viterbi_correct_count}/{n} = {100 * viterbi_correct_count / n:.1f}%")
    print(f"Потолок (верный номер есть в выдаче):    {ceiling_hits}/{n} = {100 * ceiling_hits / n:.1f}%")
    print(f"Без обнаруженного кропа таблички:        {no_crop}/{n} = {100 * no_crop / n:.1f}%")

    out_path = Path("eval_mismatches.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("file,ground_truth,geo,normalized,note\n")
        for row in mismatches:
            f.write(",".join(str(x).replace(",", ";") for x in row) + "\n")
    print(f"\nРасхождения сохранены в {out_path} ({len(mismatches)} шт.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--test-dir", default="test")
    args = parser.parse_args()
    run_eval(args.test_dir, sample=args.sample)
