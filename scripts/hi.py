from pathlib import Path

DATA_DIR = Path("data")

def count_images():
    if not DATA_DIR.exists():
        print(f"Directory '{DATA_DIR}' not found. Run the cropping script first.")
        return

    for split in ["train", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"Directory '{split_dir}' does not exist.")
            continue

        print(f"\n--- {split.upper()} DATASET ---")
        total_split = 0
        
        # Scan through each defect folder
        for class_folder in sorted(split_dir.iterdir()):
            if class_folder.is_dir():
                # Count common image extensions (.jpg, .jpeg, .png)
                count = len(list(class_folder.glob("*.jpg"))) + \
                        len(list(class_folder.glob("*.jpeg"))) + \
                        len(list(class_folder.glob("*.png")))
                total_split += count
                print(f"  {class_folder.name:<12}: {count:,} images")

        print(f"Total {split} images: {total_split:,}")

if __name__ == "__main__":
    count_images()