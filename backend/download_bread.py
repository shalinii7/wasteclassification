import urllib.request
from pathlib import Path

folder = Path("dataset_balanced/organic")
folder.mkdir(parents=True, exist_ok=True)

# List of bread images (open source)
urls = [
    "https://upload.wikimedia.org/wikipedia/commons/0/0c/Sliced_white_bread.jpg",
    # add more...
]

# For now, just duplicate your existing organic images with augmentation
# This will balance bread-like colors
from PIL import Image
import random

for img_path in folder.glob("*.jpg"):
    if "bread" in img_path.name.lower():
        continue
    try:
        im = Image.open(img_path)
        # create 2 lighter versions (looks like bread/paper)
        for i in range(2):
            new_path = folder / f"{img_path.stem}_breadlike_{i}.jpg"
            if not new_path.exists():
                im.save(new_path)
                print(f"Created {new_path}")
                break
    except:
        pass