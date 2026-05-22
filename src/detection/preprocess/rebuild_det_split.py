import json
import os
import random
from collections import defaultdict

SEED = 42
random.seed(SEED)

SRC_JSON = "data/detection_1/train/_annotations_7cls.coco.json"
OUT_TRAIN_JSON = "data/detection_1/train/_annotations_7cls_split_train.coco.json"
OUT_VAL_JSON = "data/detection_1/valid/_annotations_7cls_split_val.coco.json"

NUM_CLASSES = 7
VAL_RATIO = 0.2
MIN_VAL_BOX_PER_CLASS = 25  # có thể tăng 30 nếu dữ liệu đủ


def main():
    with open(SRC_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    anns = coco["annotations"]
    categories = coco["categories"]

    image_by_id = {img["id"]: img for img in images}
    anns_by_image = defaultdict(list)
    class_count_all = defaultdict(int)

    for ann in anns:
        anns_by_image[ann["image_id"]].append(ann)
        class_count_all[int(ann["category_id"])] += 1

    image_ids = list(image_by_id.keys())
    random.shuffle(image_ids)

    target_val_size = int(len(image_ids) * VAL_RATIO)

    # Greedy chọn val để đủ MIN_VAL_BOX_PER_CLASS
    val_ids = set()
    val_class_counts = defaultdict(int)

    # B1: ưu tiên phủ đủ lớp
    for cls in range(NUM_CLASSES):
        needed = MIN_VAL_BOX_PER_CLASS
        if class_count_all[cls] < needed:
            needed = max(1, int(class_count_all[cls] * 0.2))

        while val_class_counts[cls] < needed:
            picked = None
            for img_id in image_ids:
                if img_id in val_ids:
                    continue
                img_anns = anns_by_image.get(img_id, [])
                has_cls = any(int(a["category_id"]) == cls for a in img_anns)
                if has_cls:
                    picked = img_id
                    break
            if picked is None:
                break
            val_ids.add(picked)
            for a in anns_by_image.get(picked, []):
                val_class_counts[int(a["category_id"])] += 1

    # B2: fill tới đủ ratio
    for img_id in image_ids:
        if len(val_ids) >= target_val_size:
            break
        if img_id in val_ids:
            continue
        val_ids.add(img_id)
        for a in anns_by_image.get(img_id, []):
            val_class_counts[int(a["category_id"])] += 1

    train_ids = [i for i in image_ids if i not in val_ids]
    val_ids = list(val_ids)

    # Build train/val images
    train_images = [image_by_id[i] for i in train_ids]
    val_images = [image_by_id[i] for i in val_ids]

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    train_anns = [a for a in anns if a["image_id"] in train_id_set]
    val_anns = [a for a in anns if a["image_id"] in val_id_set]

    # Summary
    train_cls = defaultdict(int)
    val_cls = defaultdict(int)
    for a in train_anns:
        train_cls[int(a["category_id"])] += 1
    for a in val_anns:
        val_cls[int(a["category_id"])] += 1

    print("=== Split summary ===")
    print(f"Total images: {len(images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images:   {len(val_images)}")
    print("Train class box counts:", {c: train_cls[c] for c in range(NUM_CLASSES)})
    print("Val class box counts:  ", {c: val_cls[c] for c in range(NUM_CLASSES)})

    train_coco = {
        "images": train_images,
        "annotations": train_anns,
        "categories": categories,
    }
    val_coco = {
        "images": val_images,
        "annotations": val_anns,
        "categories": categories,
    }

    with open(OUT_TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    with open(OUT_VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False)

    print(f"Saved: {OUT_TRAIN_JSON}")
    print(f"Saved: {OUT_VAL_JSON}")


if __name__ == "__main__":
    main()