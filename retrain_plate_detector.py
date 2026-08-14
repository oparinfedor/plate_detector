"""Переобучение детектора таблички на честном train/val split (prepare_split.py)
с исправленными аугментациями (аудит, находка 1.3: поворот был выключен,
mosaic/erasing включены, хотя объект в кадре всегда один).

Запуск: python retrain_plate_detector.py
Веса: runs/detect/plate_honest/weights/best.pt
"""
from ultralytics import YOLO

from prepare_split import group_images, split_groups, write_split
from core.eval import run_eval

DATA_YAML = "train/data_honest.yaml"


def main():
    groups = group_images()
    train_images, val_images, train_ids, val_ids = split_groups(groups)
    assert train_ids.isdisjoint(val_ids)
    write_split(train_images, val_images)
    print(f"Split: {len(train_ids)} табличек train / {len(val_ids)} табличек val")

    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=50,
        batch=8,
        imgsz=640,
        device="cpu",
        seed=0,
        name="plate_honest",
        # находка 1.3: табличку снимают под произвольным углом - поворот
        # нужен; mosaic/erasing вредят, когда в кадре всегда один объект
        degrees=15,
        mosaic=0,
        erasing=0,
    )
    # Не угадываем путь сохранения (ultralytics сам решает, куда класть
    # веса, с учётом глобальных настроек runs_dir) - берём его из трейнера.
    best_weights = str(model.trainer.best)
    print(f"\nВеса сохранены: {best_weights}")

    metrics = model.val(data=DATA_YAML, split="val")
    print("\n=== Честная валидация (никогда не виденные фото) ===")
    print(f"mAP50:    {metrics.box.map50:.3f}  (в отчёте старой модели было 0.995 - но на train==val)")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    print(f"\n=== Сквозная точность на test/ с новыми весами ({best_weights}) ===")
    run_eval("test", plate_model_path=best_weights)


if __name__ == "__main__":
    main()
