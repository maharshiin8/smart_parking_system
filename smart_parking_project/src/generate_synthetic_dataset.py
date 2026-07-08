"""
generate_synthetic_dataset.py
------------------------------
Generates a synthetic dataset of parking-spot patches labelled
'empty' or 'occupied', so the classifier can be trained and the
whole pipeline demonstrated without needing to download a large
real-world dataset (PKLot / CNRPark-EXT) first.

This is a STAND-IN for a real dataset. For an actual college
submission, swap this out for PKLot (https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
or CNRPark-EXT and point `dataset.py` at the same folder structure:

    data/dataset/train/empty/*.jpg
    data/dataset/train/occupied/*.jpg
    data/dataset/val/empty/*.jpg
    data/dataset/val/occupied/*.jpg

Usage:
    python src/generate_synthetic_dataset.py --n_train 800 --n_val 200
"""
import argparse
import os
import random
from PIL import Image, ImageDraw

IMG_SIZE = 64


def random_asphalt(draw, size, base=(70, 70, 75)):
    w, h = size
    for _ in range(30):
        x, y = random.randint(0, w), random.randint(0, h)
        r = random.randint(1, 4)
        shade = random.randint(-12, 12)
        color = tuple(max(0, min(255, c + shade)) for c in base)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def draw_empty_spot(size=IMG_SIZE):
    img = Image.new("RGB", (size, size), (78, 78, 82))
    d = ImageDraw.Draw(img)
    random_asphalt(d, (size, size))
    # faint white parking lines on two sides
    line_color = (200, 200, 195)
    if random.random() > 0.3:
        d.line([(2, 0), (2, size)], fill=line_color, width=2)
    if random.random() > 0.3:
        d.line([(size - 2, 0), (size - 2, size)], fill=line_color, width=2)
    return img


def draw_occupied_spot(size=IMG_SIZE):
    img = draw_empty_spot(size)
    d = ImageDraw.Draw(img)
    car_color = random.choice(
        [(180, 30, 30), (30, 60, 160), (210, 210, 210), (25, 25, 25), (40, 120, 60), (200, 170, 20)]
    )
    margin = random.randint(4, 8)
    body = [margin, margin + 6, size - margin, size - margin - 6]
    d.rounded_rectangle(body, radius=8, fill=car_color, outline=(15, 15, 15))
    # windshield
    d.rectangle(
        [margin + 8, margin + 10, size - margin - 8, margin + 22],
        fill=(150, 190, 210),
    )
    return img


def build_split(root, n_empty, n_occupied):
    empty_dir = os.path.join(root, "empty")
    occ_dir = os.path.join(root, "occupied")
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(occ_dir, exist_ok=True)
    for i in range(n_empty):
        draw_empty_spot().save(os.path.join(empty_dir, f"empty_{i:04d}.jpg"), quality=90)
    for i in range(n_occupied):
        draw_occupied_spot().save(os.path.join(occ_dir, f"occupied_{i:04d}.jpg"), quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dataset")
    parser.add_argument("--n_train", type=int, default=800, help="images per class, train split")
    parser.add_argument("--n_val", type=int, default=200, help="images per class, val split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    build_split(os.path.join(args.out, "train"), args.n_train, args.n_train)
    build_split(os.path.join(args.out, "val"), args.n_val, args.n_val)
    print(f"Synthetic dataset created at {args.out}")
    print(f"  train: {args.n_train} empty + {args.n_train} occupied")
    print(f"  val:   {args.n_val} empty + {args.n_val} occupied")


if __name__ == "__main__":
    main()
