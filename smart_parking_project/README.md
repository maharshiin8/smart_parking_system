# Smart Parking System — AI/ML Computer-Vision Project

A college-level smart parking system that detects, per parking spot, whether
it is **empty** or **occupied** directly from camera images, using a
PyTorch CNN (ResNet-18 transfer learning), and displays live results on a
Flask web dashboard.

See `Smart_Parking_System_Report.docx` for the full write-up (abstract,
literature review, architecture, methodology, results, limitations, and
future work) — this README only covers how to run the code.

## 1. How it works

1. **Calibrate once per camera** — mark every parking space's pixel bounding
   box in `spots.json` (already done for the included demo lot image).
2. **Classify** — crop each spot out of the current frame and run it through
   a CNN fine-tuned to output `empty` or `occupied` with a confidence score.
3. **Aggregate & display** — combine every spot's prediction into lot-wide
   counts and show them, plus a color-coded map, on a live dashboard.

## 2. Project layout

```
smart_parking_project/
├── requirements.txt
├── Smart_Parking_System_Report.docx   # full project report
├── src/
│   ├── generate_synthetic_dataset.py  # builds a demo train/val dataset
│   ├── generate_demo_lot.py           # builds a demo full-lot camera frame
│   ├── dataset.py                     # PyTorch DataLoaders
│   ├── model.py                       # ResNet-18 transfer-learning classifier
│   ├── train.py                       # training script -> models/best_model.pt
│   └── infer.py                       # run the trained model on a full frame
├── data/
│   ├── dataset/train|val/{empty,occupied}/   # patch-classification dataset
│   └── raw/lot_camera_1.jpg, spots.json      # demo full-lot frame + spot map
├── models/
│   └── best_model.pt                  # trained checkpoint (after training)
└── app/
    ├── app.py                         # Flask backend (dashboard + API)
    ├── templates/index.html
    └── static/ (style.css, script.js, status.json, images)
```

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on pretrained weights:** `model.py` tries to download ImageNet
> weights for the ResNet-18 backbone the first time you train. This
> requires internet access. If it's unavailable, the code automatically
> falls back to random initialization so the pipeline still runs — but for
> real accuracy, training with `pretrained=True` (the default) on a machine
> with internet access is strongly recommended.

## 4. Quick start — end-to-end demo

```bash
# 1. Generate a demo dataset (or replace with a real dataset — see below)
python src/generate_synthetic_dataset.py --n_train 500 --n_val 120

# 2. Generate a demo full-lot camera frame + spot map
python src/generate_demo_lot.py --rows 3 --cols 6

# 3. Train the classifier
python src/train.py --epochs 8 --data_root ../data/dataset --out_dir ../models
#   (run from src/, or pass --data_root data/dataset --out_dir models from project root)

# 4. Run inference on the demo lot frame
python src/infer.py --image data/raw/lot_camera_1.jpg --spots data/raw/spots.json \
    --model models/best_model.pt --out_json app/static/status.json \
    --out_image app/static/annotated.jpg

# 5. Launch the dashboard
cd app
python app.py
# open http://localhost:5000
```

Click **"Pull new frame"** on the dashboard to simulate a new camera frame
arriving (randomized occupancy) and watch the CNN reclassify every spot
live.

## 5. Using a real dataset (recommended for real accuracy numbers)

Swap the synthetic dataset for a real one, e.g.
[PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) or
CNRPark-EXT. Reorganize the images into:

```
data/dataset/train/empty/*.jpg
data/dataset/train/occupied/*.jpg
data/dataset/val/empty/*.jpg
data/dataset/val/occupied/*.jpg
```

then run `src/train.py` as above — no other code changes are required.

## 6. Using a real camera feed

Replace the `simulate_new_frame()` call in `app/app.py`'s `/api/refresh`
route with a frame grab from your camera (e.g. via OpenCV's
`cv2.VideoCapture` on an RTSP URL), keep `spots.json` calibrated for that
camera's fixed viewpoint, and the rest of the pipeline (crop → classify →
aggregate → dashboard) works unchanged.

## 7. Tech stack

Python · PyTorch / torchvision (ResNet-18 transfer learning) · Pillow ·
Flask · HTML/CSS/JavaScript
