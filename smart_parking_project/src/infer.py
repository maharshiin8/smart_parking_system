"""
infer.py
--------
Runs the trained classifier over every parking spot in a camera frame.

Given:
  - a full-lot image (e.g. from a fixed CCTV/IP camera)
  - a spots.json describing each spot's bounding box within that image

Produces:
  - status.json: per-spot prediction + confidence + summary counts
  - annotated.jpg: the lot image with green/red boxes drawn over each spot

Usage:
    python src/infer.py \
        --image data/raw/lot_camera_1.jpg \
        --spots data/raw/spots.json \
        --model models/best_model.pt \
        --out_json app/static/status.json \
        --out_image app/static/annotated.jpg
"""
import argparse
import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont

from model import build_model
from dataset import eval_transform


def load_model(model_path, device="cpu"):
    ckpt = torch.load(model_path, map_location=device)
    model = build_model(device=device, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]


def classify_spots(image_path, spots_path, model, classes, device="cpu"):
    img = Image.open(image_path).convert("RGB")
    with open(spots_path) as f:
        meta = json.load(f)

    results = []
    for spot in meta["spots"]:
        x1, y1, x2, y2 = spot["bbox"]
        patch = img.crop((x1, y1, x2, y2))
        tensor = eval_transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])

        results.append({
            "id": spot["id"],
            "bbox": spot["bbox"],
            "prediction": classes[pred_idx],
            "confidence": round(confidence, 4),
        })

    return img, meta, results


def draw_annotations(img, results):
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = (40, 200, 90) if r["prediction"] == "empty" else (220, 60, 60)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{r['id']}"
        draw.rectangle([x1, y1, x1 + 34, y1 + 14], fill=color)
        draw.text((x1 + 3, y1 + 1), label, fill=(255, 255, 255))
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/raw/lot_camera_1.jpg")
    parser.add_argument("--spots", default="data/raw/spots.json")
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--out_json", default="app/static/status.json")
    parser.add_argument("--out_image", default="app/static/annotated.jpg")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(args.model, device)
    img, meta, results = classify_spots(args.image, args.spots, model, classes, device)

    empty_count = sum(1 for r in results if r["prediction"] == "empty")
    occupied_count = len(results) - empty_count

    # accuracy vs synthetic ground truth, if present (demo/eval only)
    gt_available = all("ground_truth" in s for s in meta["spots"])
    accuracy = None
    if gt_available:
        gt_map = {s["id"]: s["ground_truth"] for s in meta["spots"]}
        correct = sum(1 for r in results if r["prediction"] == gt_map[r["id"]])
        accuracy = round(correct / len(results), 4)

    output = {
        "image": meta["image"],
        "total_spots": len(results),
        "empty": empty_count,
        "occupied": occupied_count,
        "occupancy_rate": round(occupied_count / len(results), 4),
        "spots": results,
        "demo_accuracy_vs_ground_truth": accuracy,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)

    annotated = draw_annotations(img, results)
    os.makedirs(os.path.dirname(args.out_image), exist_ok=True)
    annotated.save(args.out_image, quality=92)

    print(f"Spots: {len(results)} | empty: {empty_count} | occupied: {occupied_count}")
    if accuracy is not None:
        print(f"Demo accuracy vs synthetic ground truth: {accuracy*100:.1f}%")
    print(f"Wrote {args.out_json} and {args.out_image}")


if __name__ == "__main__":
    main()
