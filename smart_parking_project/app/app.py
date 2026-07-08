"""
app.py
------
Flask backend for the Smart Parking dashboard.

Endpoints:
    GET  /                -> dashboard page
    GET  /api/status      -> current per-spot occupancy status (JSON)
    POST /api/refresh     -> simulate a new camera frame (randomized
                              occupancy), run the trained CNN over every
                              spot, and return the updated status

In a real deployment, /api/refresh would instead pull the newest frame
from an IP camera / RTSP stream and run the same classify_spots() call
on it. The simulation here exists so the dashboard can be demoed live
without physical camera hardware.
"""
import os
import sys
import json
import random
import time

from flask import Flask, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model            # noqa: E402
from dataset import eval_transform       # noqa: E402
from infer import classify_spots, draw_annotations, load_model  # noqa: E402
from generate_demo_lot import main as _unused  # noqa: F401  (keeps module importable)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pt")
SPOTS_JSON = os.path.join(STATIC_DIR, "spots.json")
LOT_IMAGE = os.path.join(STATIC_DIR, "lot_camera_1.jpg")
STATUS_JSON = os.path.join(STATIC_DIR, "status.json")
ANNOTATED_IMAGE = os.path.join(STATIC_DIR, "annotated.jpg")

app = Flask(__name__)

_device = "cpu"
_model, _classes = load_model(MODEL_PATH, _device)


def run_inference_and_cache():
    img, meta, results = classify_spots(LOT_IMAGE, SPOTS_JSON, _model, _classes, _device)
    empty_count = sum(1 for r in results if r["prediction"] == "empty")
    occupied_count = len(results) - empty_count

    output = {
        "image": "lot_camera_1.jpg",
        "annotated_image": "annotated.jpg",
        "total_spots": len(results),
        "empty": empty_count,
        "occupied": occupied_count,
        "occupancy_rate": round(occupied_count / len(results), 4),
        "spots": results,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(STATUS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    annotated = draw_annotations(img, results)
    annotated.save(ANNOTATED_IMAGE, quality=92)
    return output


def simulate_new_frame():
    """
    Randomly repaints some spots occupied/empty in the lot image to mimic
    a fresh camera frame arriving, then re-runs the CNN on it. This is
    what lets the dashboard feel "live" in a demo/offline setting.
    """
    from PIL import Image, ImageDraw

    with open(SPOTS_JSON) as f:
        meta = json.load(f)

    img = Image.new("RGB", tuple(meta["image_size"]), (60, 60, 64))
    draw = ImageDraw.Draw(img)
    for _ in range(600):
        x = random.randint(0, meta["image_size"][0])
        y = random.randint(0, meta["image_size"][1])
        r = random.randint(1, 3)
        shade = random.randint(-10, 10)
        c = tuple(max(0, min(255, 60 + shade)) for _ in range(3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)

    car_colors = [(180, 30, 30), (30, 60, 160), (210, 210, 210), (25, 25, 25),
                  (40, 120, 60), (200, 170, 20), (140, 90, 40)]

    for spot in meta["spots"]:
        x1, y1, x2, y2 = spot["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(210, 210, 200), width=2)
        occupied = random.random() < 0.55
        spot["ground_truth"] = "occupied" if occupied else "empty"
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

    img.save(LOT_IMAGE, quality=92)
    with open(SPOTS_JSON, "w") as f:
        json.dump(meta, f, indent=2)


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    if not os.path.exists(STATUS_JSON):
        run_inference_and_cache()
    with open(STATUS_JSON) as f:
        return jsonify(json.load(f))


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    simulate_new_frame()
    output = run_inference_and_cache()
    return jsonify(output)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    run_inference_and_cache()
    # debug/reloader is intentionally off: the app writes status.json and
    # annotated.jpg into the static/ folder on every refresh, which would
    # otherwise trigger the reloader to restart the server mid-request.
    app.run(host="0.0.0.0", port=5000, debug=False)
