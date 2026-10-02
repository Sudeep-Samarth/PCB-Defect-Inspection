# Step1
'''
1. Created folders for all classes in /data folder
2. Converted all annotations in each .txt file into tuples, for each board
3. Cropped and saved all defects' images in data folder
'''

import os
import cv2
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

# Label mapping for DeepPCB class indices (1 to 6)
LABEL_MAP = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin_hole"
}

RAW_DATASET_DIR = Path("raw_dataset/PCBData")
OUTPUT_BASE_DIR = Path("data")
PADDING_RATIO = 0.15

"""Ensure category subdirectories exist in both train and test splits."""
def setup_output_dirs():
    for split in ["train", "test"]:
        for class_name in LABEL_MAP.values():
            (OUTPUT_BASE_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


"""Parses bounding boxes and class types from an annotation text file."""
def parse_annotation_file(txt_path):
    boxes=[]
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >=5:
                x1,y1,x2,y2,defect_type = map(int, parts[:5])
                boxes.append((x1,y1,x2,y2,defect_type))
    return boxes


"""Crops a defect patch with contextual padding and saves it."""
def crop_and_save(img, box, output_path):
    h, w = img.shape[:2]
    x1,y1,x2,y2,defect_type = box

    bw = x2-x1
    bh = y2-y1

    pad_w = int(bw* PADDING_RATIO)
    pad_h = int(bh* PADDING_RATIO)

    # Expand coordinates while remaining inside image boundaries
    nx1 = max(0, x1-pad_w)
    ny1 = max(0, y1-pad_h)
    nx2 = min(w, x2+pad_w)
    ny2 = min(h, y2+pad_h)

    crop = img[ny1:ny2, nx1:nx2]
    if crop.size > 0:
        cv2.imwrite(str(output_path), crop)


""" Locates the corresponding test image for an annotation file across sibling folders.
    Handles DeepPCB's structure: groupXXXXX/XXXXX_not/XXXXXxxx.txt -> groupXXXXX/XXXXX/XXXXXxxx_test.jpg """
def locate_test_image(ann_path): # [ann_path = say 00041000.txt]
    # we have to find 00041000_test.jpg
    board_id = ann_path.stem.replace("_not", "")
    parent_dir = ann_path.parent # [00041_not]

    # 1. Sibling image directory (removing '_not' suffix from parent directory)
    if parent_dir.name.endswith("_not"):
        sibling_img_dir = parent_dir.parent / parent_dir.name[:-4] # [group00041/00041]
        candidate = sibling_img_dir / f"{board_id}_test.jpg"
        if candidate.exists():
            return candidate

    # 2. Check same directory fallback
    candidate = parent_dir / f"{board_id}_test.jpg" # [00041_not/00041_test.jpg] (not in our case but just to be sure)
    if candidate.exists():
        return candidate

    # 3. Fuzzy search for board_id containing 'test'
    candidates = list(parent_dir.parent.glob(f"**/{board_id}_test.jpg")) # [group00041/00041_test.jpg] (not in our case but just to be sure)
    if candidates:
        return candidates[0]

    return None

    


def process_dataset(test_ratio=0.2, random_state = 42):
    setup_output_dirs() # [data/train/open, etc; data/test/open, etc]

    ann_files = list(RAW_DATASET_DIR.glob("**/*_not/*.txt")) # [ [PCBData/group00041/00041_not/00041000.txt, .../00041001.txt, etc] ]
    if not ann_files:
        # Fallback if names differ: look for all .txt files
        ann_files = [f for f in RAW_DATASET_DIR.glob("**/*.txt") if "trainval" not in f.name and "test" not in f.name]

    print(f"Found {len(ann_files)} annotation files.")
    if len(ann_files) == 0:
        print("Error: No annotation files found. Check your RAW_DATASET_DIR path.")
        return

    # Split dataset at the image/board level to prevent data leakage
    train_files, test_files = train_test_split(ann_files, test_size = test_ratio, random_state=random_state)

    splits = [("train", train_files), ("test", test_files)]
    stats = {
        "train" : {name: 0 for name in LABEL_MAP.values()},
        "test": {name: 0 for name in LABEL_MAP.values()}
    }

    for split_name, file_list in splits:
        print(f"\nProcessing {split_name} split ({len(file_list)} boards)...")
        for ann_path in file_list: # [say 00041000.txt]
            test_img_path = locate_test_image(ann_path)
            if test_img_path is None or not test_img_path.exists():
                continue

            img = cv2.imread(str(test_img_path))
            if img is None:
                continue

            boxes = parse_annotation_file(ann_path)
            for idx, box in enumerate(boxes):
                _, _, _, _, defect_type = box
                if defect_type in LABEL_MAP:
                    class_name = LABEL_MAP[defect_type]
                    out_filename = f"{test_img_path.stem}_defect_{idx}_{class_name}.jpg"
                    out_file_path = OUTPUT_BASE_DIR / split_name / class_name / out_filename

                    crop_and_save(img, box, out_file_path)
                    stats[split_name][class_name] += 1


    print("\n--- Cropping Complete ---")
    for split_name in ["train", "test"]:
        print(f"\n{split_name.upper()} counts: ")
        for defect_name, count in stats[split_name].items():
            print(f"  {defect_name:12}: {count}")


if __name__ == "__main__":
    process_dataset()


