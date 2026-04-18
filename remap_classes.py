"""
Remaps TACO dataset from 9 classes to 3 classes.

Run this ONCE before training any model.

Class mapping:
  Recyclable     (1 / YOLO 0): Bottle(1), Glass(4), Metal(5), Paper(7), Plastic(8)
  Non-recyclable (2 / YOLO 1): Cigarette(2), Foam(3)
  Other          (3 / YOLO 2): Other(6), Unlabeled(9)

Actions:
  1. Patches archive/dataset_v2/{train,val,test}_annotations.json in-place
  2. Regenerates archive/dataset_v2/labels/{train,val,test}/*.txt (YOLO seg format)
  3. Updates taco_rtdetr.yaml  ->  nc: 3, names: [Recyclable, Non-recyclable, Other]
"""

import json
import os
import yaml
from pathlib import Path
from collections import defaultdict

BASE_DIR    = r"C:\Users\User\Desktop\Ipynb"
DATASET_DIR = os.path.join(BASE_DIR, "archive", "dataset_v2")

# Old COCO category_id -> new COCO category_id
CAT_REMAP = {
    1: 1,  # Bottle      -> Recyclable
    2: 2,  # Cigarette   -> Non-recyclable
    3: 2,  # Foam        -> Non-recyclable
    4: 1,  # Glass       -> Recyclable
    5: 1,  # Metal       -> Recyclable
    6: 3,  # Other       -> Other
    7: 1,  # Paper       -> Recyclable
    8: 1,  # Plastic     -> Recyclable
    9: 3,  # Unlabeled   -> Other
}

NEW_CATEGORIES = [
    {"id": 1, "name": "Recyclable",     "supercategory": "waste"},
    {"id": 2, "name": "Non-recyclable", "supercategory": "waste"},
    {"id": 3, "name": "Other",          "supercategory": "waste"},
]


def remap_coco_json(ann_file):
    with open(ann_file) as f:
        data = json.load(f)

    data["categories"] = NEW_CATEGORIES
    for ann in data["annotations"]:
        ann["category_id"] = CAT_REMAP[ann["category_id"]]

    with open(ann_file, "w") as f:
        json.dump(data, f)

    n_ann = len(data["annotations"])
    print(f"  Patched {n_ann} annotations -> {ann_file}")


def regenerate_yolo_labels(ann_file, label_dir):
    """Write YOLO segmentation labels (polygon format) from a COCO JSON."""
    os.makedirs(label_dir, exist_ok=True)

    with open(ann_file) as f:
        data = json.load(f)

    img_info    = {img["id"]: img for img in data["images"]}
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    skipped = 0
    for img_id, img in img_info.items():
        iw, ih = img["width"], img["height"]
        stem   = Path(img["file_name"]).stem
        lines  = []

        for ann in anns_by_img[img_id]:
            cat  = ann["category_id"] - 1   # COCO 1-indexed -> YOLO 0-indexed
            segs = ann.get("segmentation", [])

            if segs and isinstance(segs, list) and len(segs[0]) >= 6:
                seg    = segs[0]
                coords = []
                for i in range(0, len(seg) - 1, 2):
                    coords.append(max(0.0, min(1.0, seg[i]     / iw)))
                    coords.append(max(0.0, min(1.0, seg[i + 1] / ih)))
            else:
                x, y, bw, bh = ann["bbox"]
                if bw <= 0 or bh <= 0:
                    skipped += 1
                    continue
                coords = [
                    x / iw,        y / ih,
                    (x + bw) / iw, y / ih,
                    (x + bw) / iw, (y + bh) / ih,
                    x / iw,        (y + bh) / ih,
                ]
                coords = [max(0.0, min(1.0, v)) for v in coords]

            lines.append(f"{cat} " + " ".join(f"{v:.6f}" for v in coords))

        with open(os.path.join(label_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    print(f"  Labels written to {label_dir}  (skipped {skipped} zero-size boxes)")


def update_yaml():
    yaml_path = os.path.join(BASE_DIR, "taco_rtdetr.yaml")

    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "path":  DATASET_DIR.replace("\\", "/"),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
        }

    cfg["nc"]    = 3
    cfg["names"] = ["Recyclable", "Non-recyclable", "Other"]

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"  Updated YAML -> {yaml_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("  TACO class remapping: 9 classes -> 3 classes")
    print("=" * 55)

    for split in ["train", "val", "test"]:
        ann_file  = os.path.join(DATASET_DIR, f"{split}_annotations.json")
        label_dir = os.path.join(DATASET_DIR, "labels", split)
        print(f"\n[{split}]")
        remap_coco_json(ann_file)
        regenerate_yolo_labels(ann_file, label_dir)

    print("\n[yaml]")
    update_yaml()

    print("\n" + "=" * 55)
    print("Done. Class mapping:")
    print("  1 (YOLO 0) = Recyclable     (Bottle, Glass, Metal, Paper, Plastic)")
    print("  2 (YOLO 1) = Non-recyclable (Cigarette, Foam)")
    print("  3 (YOLO 2) = Other          (Other, Unlabeled)")
    print("=" * 55)
