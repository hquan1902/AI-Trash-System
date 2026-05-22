import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


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


MAP_D2_TO_7 = {
    ".dot": "trash",
    "Cardboard": "cardboard",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
}


MAP_BIO_TO_7 = {
    "Organic Waste": "biological",
    "Paper Waste": "paper",
    "trash": "trash",
}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _remap_and_prefix(
    coco: dict,
    category_name_map: dict,
    file_name_prefix: str,
    next_image_id: int,
    next_ann_id: int,
    sample_max_images: int | None = None,
    sample_seed: int = 42,
):
    old_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    unknown_names = sorted(set(old_id_to_name.values()) - set(category_name_map.keys()))
    if unknown_names:
        raise ValueError(f"Unknown categories for mapping: {unknown_names}")

    old_image_to_new_image = {}
    selected_old_image_ids = None
    if sample_max_images is not None and sample_max_images > 0:
        all_img_ids = [img["id"] for img in coco.get("images", [])]
        if sample_max_images < len(all_img_ids):
            rnd = random.Random(sample_seed)
            selected_old_image_ids = set(rnd.sample(all_img_ids, sample_max_images))

    new_images = []
    for img in coco.get("images", []):
        if selected_old_image_ids is not None and img["id"] not in selected_old_image_ids:
            continue
        new_img = dict(img)
        new_img["id"] = next_image_id
        new_img["file_name"] = f"{file_name_prefix}/{img['file_name']}".replace("\\", "/")
        old_image_to_new_image[img["id"]] = next_image_id
        next_image_id += 1
        new_images.append(new_img)

    new_annotations = []
    dropped = 0
    cls_counts = Counter()

    for ann in coco.get("annotations", []):
        old_cat_id = ann.get("category_id")
        old_name = old_id_to_name.get(old_cat_id)
        mapped_name = category_name_map.get(old_name)
        if mapped_name is None:
            dropped += 1
            continue

        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            dropped += 1
            continue
        _, _, w, h = bbox
        if w <= 0 or h <= 0:
            dropped += 1
            continue

        old_img_id = ann.get("image_id")
        if old_img_id not in old_image_to_new_image:
            dropped += 1
            continue

        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = old_image_to_new_image[old_img_id]
        new_ann["category_id"] = TARGET_ID[mapped_name]
        next_ann_id += 1

        new_annotations.append(new_ann)
        cls_counts[mapped_name] += 1

    return {
        "images": new_images,
        "annotations": new_annotations,
        "dropped": dropped,
        "next_image_id": next_image_id,
        "next_ann_id": next_ann_id,
        "class_counts": cls_counts,
    }


def _class_aware_split_train_val(coco_train_merged: dict, val_ratio: float, min_val_boxes_per_class: int, seed: int):
    random.seed(seed)

    images = coco_train_merged["images"]
    anns = coco_train_merged["annotations"]
    categories = coco_train_merged["categories"]

    image_by_id = {img["id"]: img for img in images}
    anns_by_image = defaultdict(list)
    class_count_all = defaultdict(int)
    for ann in anns:
        anns_by_image[ann["image_id"]].append(ann)
        class_count_all[int(ann["category_id"])] += 1

    image_ids = list(image_by_id.keys())
    random.shuffle(image_ids)

    target_val_size = max(1, int(len(image_ids) * val_ratio))
    val_ids = set()
    val_class_counts = defaultdict(int)

    for cls in range(len(TARGET_CLASSES)):
        needed = min_val_boxes_per_class
        if class_count_all[cls] < needed:
            needed = max(1, int(class_count_all[cls] * 0.2))

        while val_class_counts[cls] < needed:
            picked = None
            for img_id in image_ids:
                if img_id in val_ids:
                    continue
                img_anns = anns_by_image.get(img_id, [])
                if any(int(a["category_id"]) == cls for a in img_anns):
                    picked = img_id
                    break
            if picked is None:
                break
            val_ids.add(picked)
            for a in anns_by_image.get(picked, []):
                val_class_counts[int(a["category_id"])] += 1

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

    train_set = set(train_ids)
    val_set = set(val_ids)

    out_train = {
        "images": [image_by_id[i] for i in train_ids],
        "annotations": [a for a in anns if a["image_id"] in train_set],
        "categories": categories,
    }
    out_val = {
        "images": [image_by_id[i] for i in val_ids],
        "annotations": [a for a in anns if a["image_id"] in val_set],
        "categories": categories,
    }
    return out_train, out_val


def _count_by_class(coco: dict):
    id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    ctr = Counter({name: 0 for name in TARGET_CLASSES})
    for ann in coco.get("annotations", []):
        ctr[id_to_name[int(ann["category_id"])]] += 1
    return dict(ctr)


def main():
    parser = argparse.ArgumentParser(description="Merge detection_1 + detection_2 into unified 7-class dataset")
    parser.add_argument("--data-root", default="data", help="Root folder containing detection_1 and detection_2")
    parser.add_argument("--out-root", default="data/detection_merged", help="Output merged dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio from merged train pool")
    parser.add_argument("--min-val-boxes-per-class", type=int, default=40, help="Minimum validation boxes per class")
    parser.add_argument("--bio-train-max-images", type=int, default=1800, help="Random cap for biological_class/train images (<=0 means use all)")
    parser.add_argument("--bio-valid-max-images", type=int, default=300, help="Random cap for biological_class/valid images (<=0 means use all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    train_sources = [
        ("detection_1", "train", "_annotations.coco.json", MAP_25_TO_7),
        ("detection_2", "train", "_annotations.coco.json", MAP_D2_TO_7),
        ("biological_class", "train", "_annotations.coco.json", MAP_BIO_TO_7),
        ("biological_class", "valid", "_annotations.coco.json", MAP_BIO_TO_7),
    ]
    test_sources = [
        ("detection_1", "test", "_annotations.coco.json", MAP_25_TO_7),
        ("detection_2", "test", "_annotations.coco.json", MAP_D2_TO_7),
    ]

    next_image_id = 1
    next_ann_id = 1
    merged_train_images = []
    merged_train_annotations = []

    print("[1/3] Remap + merge train pools")
    for source_idx, (ds, split, json_name, name_map) in enumerate(train_sources):
        path = data_root / ds / split / json_name
        coco = _load_json(path)

        sample_max_images = None
        if ds == "biological_class" and split == "train":
            sample_max_images = args.bio_train_max_images if args.bio_train_max_images > 0 else None
        elif ds == "biological_class" and split == "valid":
            sample_max_images = args.bio_valid_max_images if args.bio_valid_max_images > 0 else None

        remapped = _remap_and_prefix(
            coco=coco,
            category_name_map=name_map,
            file_name_prefix=f"{ds}/{split}",
            next_image_id=next_image_id,
            next_ann_id=next_ann_id,
            sample_max_images=sample_max_images,
            sample_seed=args.seed + source_idx,
        )
        next_image_id = remapped["next_image_id"]
        next_ann_id = remapped["next_ann_id"]

        merged_train_images.extend(remapped["images"])
        merged_train_annotations.extend(remapped["annotations"])
        print(f"  - {ds}/{split}: images={len(remapped['images'])}, anns={len(remapped['annotations'])}, dropped={remapped['dropped']}, cls={dict(remapped['class_counts'])}")

    categories = [{"id": i, "name": name} for i, name in enumerate(TARGET_CLASSES)]
    merged_train_coco = {
        "images": merged_train_images,
        "annotations": merged_train_annotations,
        "categories": categories,
    }

    print("[2/3] Build class-aware train/val split")
    out_train, out_val = _class_aware_split_train_val(
        merged_train_coco,
        val_ratio=args.val_ratio,
        min_val_boxes_per_class=args.min_val_boxes_per_class,
        seed=args.seed,
    )

    train_json = out_root / "train" / "_annotations_7cls_split_train.coco.json"
    val_json = out_root / "valid" / "_annotations_7cls_split_val.coco.json"
    _save_json(train_json, out_train)
    _save_json(val_json, out_val)

    print(f"  - train saved: {train_json} | images={len(out_train['images'])}, anns={len(out_train['annotations'])}, cls={_count_by_class(out_train)}")
    print(f"  - val saved  : {val_json} | images={len(out_val['images'])}, anns={len(out_val['annotations'])}, cls={_count_by_class(out_val)}")

    print("[3/3] Remap + merge test pools")
    merged_test_images = []
    merged_test_annotations = []
    for ds, split, json_name, name_map in test_sources:
        path = data_root / ds / split / json_name
        coco = _load_json(path)
        remapped = _remap_and_prefix(
            coco=coco,
            category_name_map=name_map,
            file_name_prefix=f"{ds}/{split}",
            next_image_id=next_image_id,
            next_ann_id=next_ann_id,
        )
        next_image_id = remapped["next_image_id"]
        next_ann_id = remapped["next_ann_id"]
        merged_test_images.extend(remapped["images"])
        merged_test_annotations.extend(remapped["annotations"])
        print(f"  - {ds}/{split}: images={len(remapped['images'])}, anns={len(remapped['annotations'])}, dropped={remapped['dropped']}, cls={dict(remapped['class_counts'])}")

    out_test = {
        "images": merged_test_images,
        "annotations": merged_test_annotations,
        "categories": categories,
    }
    test_json = out_root / "test" / "_annotations_7cls_merged_test.coco.json"
    _save_json(test_json, out_test)
    print(f"  - test saved : {test_json} | images={len(out_test['images'])}, anns={len(out_test['annotations'])}, cls={_count_by_class(out_test)}")

    print("\nUse these env settings:")
    print('  $env:DET_TRAIN_IMG_DIR="data"')
    print('  $env:DET_VAL_IMG_DIR="data"')
    print(f'  $env:DET_TRAIN_JSON="{train_json.as_posix()}"')
    print(f'  $env:DET_VAL_JSON="{val_json.as_posix()}"')


if __name__ == "__main__":
    main()
