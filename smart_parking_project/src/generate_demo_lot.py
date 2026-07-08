"""
generate_demo_lot.py
---------------------
Builds one full "camera view" image of a parking lot (a grid of spots,
some occupied, some empty) plus a spots.json file describing the
pixel bounding box of every spot in that image.

In a real deployment, spots.json would be created ONCE per camera by
manually marking each parking-space polygon in the camera's fixed
field of view (this is standard practice in PKLot / CNRPark-style
systems, since the camera does not move).

Usage:
    python src/generate_demo_lot.py --rows 3 --cols 6
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw

CELL = 90
MARGIN = 30
GAP = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--occupied_ratio", type=float, default=0.55)
    parser.add_argument("--out_image", default="data/raw/lot_camera_1.jpg")
    parser.add_argument("--out_json", default="data/raw/spots.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    rows, cols = args.rows, args.cols
    width = MARGIN * 2 + cols * (CELL + GAP)
    height = MARGIN * 2 + rows * (CELL + GAP) + 40  # + lane space at top

    img = Image.new("RGB", (width, height), (60, 60, 64))
    draw = ImageDraw.Draw(img)
    # asphalt texture
    for _ in range(600):
        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(1, 3)
        shade = random.randint(-10, 10)
        c = tuple(max(0, min(255, 60 + shade)) for _ in range(3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)

    car_colors = [(180, 30, 30), (30, 60, 160), (210, 210, 210), (25, 25, 25),
                  (40, 120, 60), (200, 170, 20), (140, 90, 40)]

    spots = []
    spot_id = 0
    for r in range(rows):
        for c in range(cols):
            x1 = MARGIN + c * (CELL + GAP)
            y1 = MARGIN + 40 + r * (CELL + GAP)
            x2 = x1 + CELL
            y2 = y1 + CELL

            # parking lines
            draw.rectangle([x1, y1, x2, y2], outline=(210, 210, 200), width=2)

            occupied = random.random() < args.occupied_ratio
            if occupied:
                color = random.choice(car_colors)
                pad = random.randint(6, 10)
                draw.rounded_rectangle(
                    [x1 + pad, y1 + pad + 4, x2 - pad, y2 - pad - 4],
                    radius=10, fill=color, outline=(15, 15, 15)
                )
                draw.rectangle(
                    [x1 + pad + 6, y1 + pad + 10, x2 - pad - 6, y1 + pad + 26],
                    fill=(150, 190, 210)
                )

            spot_id += 1
            spots.append({
                "id": f"S{spot_id}",
                "row": r,
                "col": c,
                "bbox": [x1, y1, x2, y2],   # [xmin, ymin, xmax, ymax]
                "ground_truth": "occupied" if occupied else "empty",  # for demo evaluation only
            })

    os.makedirs(os.path.dirname(args.out_image), exist_ok=True)
    img.save(args.out_image, quality=92)
    with open(args.out_json, "w") as f:
        json.dump({"image": os.path.basename(args.out_image),
                   "image_size": [width, height], "spots": spots}, f, indent=2)

    print(f"Saved lot image -> {args.out_image}")
    print(f"Saved spot map  -> {args.out_json}  ({len(spots)} spots)")


if __name__ == "__main__":
    main()
