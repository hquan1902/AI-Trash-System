import json
from pathlib import Path
from collections import Counter

DATA_ROOT = Path("data/detection_1")
SPLITS = ["train", "valid", "test"]
COCO_NAME = "_annotations.coco.json"

TARGET_CLASSES = [
    "biological",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]
TARGET_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

MAP_25_TO_7 = {
    "wastes-hyVU": "trash",
    "cardboard": "cardboard",
    "carton packaging": "cardboard",
    "cigarette": "trash",
    "clean paper": "paper",
    "clear plastic": "plastic",
    "contaminated paper": "paper",
    "food packaging": "trash",
    "food scraps": "biological",
    "glass": "glass",
    "medical waste": "trash",
    "metal": "metal",
    "paper bag": "paper",
    "paper cup": "paper",
    "plastic bottle": "plastic",
    "plastic container": "plastic",
    "plastic cup": "plastic",
    "plastic lid": "plastic",
    "plastic packaging": "plastic",
    "plastic utensil": "plastic",
    "printed cardboard": "cardboard",
    "sanitary waste": "trash",
    "straw": "plastic",
    "styrofoam": "plastic",
    "wood": "trash",
}


def remap_one_split(split_dir: Path):
    coco_path = split_dir / COCO_NAME
    out_path = split_dir / "_annotations_7cls.coco.json"

    if not coco_path.exists():
        raise FileNotFoundError(f"Missing COCO file: {coco_path}")

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    old_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    unknown_old_names = sorted(set(old_id_to_name.values()) - set(MAP_25_TO_7.keys()))
    if unknown_old_names:
        raise ValueError(f"[{split_dir.name}] Unmapped old categories found: {unknown_old_names}")

    old_id_to_new_id = {}
    for old_id, old_name in old_id_to_name.items():
        new_name = MAP_25_TO_7[old_name]
        old_id_to_new_id[old_id] = TARGET_ID[new_name]

    new_annotations = []
    dropped = 0
    per_new_class = Counter()

    for ann in coco["annotations"]:
        old_cat_id = ann["category_id"]
        if old_cat_id not in old_id_to_new_id:
            dropped += 1
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            dropped += 1
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = old_id_to_new_id[old_cat_id]
        new_annotations.append(new_ann)
        per_new_class[new_ann["category_id"]] += 1

    new_categories = [{"id": i, "name": name} for i, name in enumerate(TARGET_CLASSES)]
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": coco["images"],
        "annotations": new_annotations,
        "categories": new_categories,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False)

    print(f"\n[{split_dir.name}]")
    print(f"Input : {coco_path}")
    print(f"Output: {out_path}")
    print(f"Images: {len(new_coco['images'])}")
    print(f"Anns  : {len(coco['annotations'])} -> {len(new_annotations)} (dropped={dropped})")
    print("Per-class ann count:")
    for i, name in enumerate(TARGET_CLASSES):
        print(f"  {i:>2} {name:<10}: {per_new_class[i]}")


def main():
    print("Remap COCO categories: 25 -> 7")
    for split in SPLITS:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            print(f"Skip missing split dir: {split_dir}")
            continue
        remap_one_split(split_dir)

    print("\nXong, có thể train detection ở file coco *_annotations_7cls.coco.json")


if __name__ == "__main__":
    main()
