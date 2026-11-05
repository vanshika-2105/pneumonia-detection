from pathlib import Path

DATA_DIR = Path("data/raw/chest_xray")

for split in ["train", "test", "val"]:
    print(f"\n{split.upper()} SET")

    for label in ["NORMAL", "PNEUMONIA"]:
        folder = DATA_DIR / split / label
        image_count = len(list(folder.glob("*")))
        print(f"{label}: {image_count} images")
