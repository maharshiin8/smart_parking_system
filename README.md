# 🚗 Smart Parking System — AI Computer Vision Project

An AI-powered smart parking management system that detects whether individual parking spaces are **empty or occupied** using deep learning and computer vision. The system uses a **PyTorch CNN with ResNet-18 transfer learning** to classify parking spaces from camera images and displays real-time occupancy information through a **Flask web dashboard**.

This project was developed as a **college group project** to demonstrate the practical application of deep learning, computer vision, and web development for automated parking management.

---

## ✨ Features

* 🚘 Individual parking space occupancy detection
* 🧠 Deep learning classification using ResNet-18
* 📷 Camera image-based parking analysis
* 🎨 Color-coded parking availability map
* 📊 Real-time dashboard showing:

  * Total parking spaces
  * Occupied spaces
  * Available spaces
  * Occupancy percentage
* 🔄 Live status updates
* 📈 Confidence score for each prediction

---

## 🏗️ System Architecture

```text
Camera Image
      |
      ↓
Parking Space Mapping
      |
      ↓
Image Cropping
      |
      ↓
CNN Classification Model
(ResNet-18 Transfer Learning)
      |
      ↓
Occupancy Prediction
      |
      ↓
Flask Dashboard
```

---

## 🛠️ Tech Stack

### AI / Machine Learning

* Python
* PyTorch
* Torchvision
* ResNet-18 Transfer Learning
* OpenCV
* Pillow

### Backend

* Flask
* REST API
* JSON-based data handling

### Frontend

* HTML
* CSS
* JavaScript

### Tools

* Git
* GitHub

---

## 📂 Project Structure

```text
smart-parking-system/

├── app/
│   ├── app.py                  # Flask dashboard backend
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       ├── script.js
│       ├── status.json
│       └── annotated.jpg
│
├── src/
│   ├── dataset.py              # PyTorch dataset loader
│   ├── model.py                # ResNet-18 model
│   ├── train.py                # Model training
│   ├── infer.py                # Parking prediction
│   ├── generate_demo_lot.py
│   └── generate_synthetic_dataset.py
│
├── data/
│   ├── dataset/
│   └── raw/
│       ├── lot_camera_1.jpg
│       └── spots.json
│
├── models/
│   └── best_model.pt
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/smart-parking-system.git

cd smart-parking-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## 1. Generate Training Dataset

```bash
python src/generate_synthetic_dataset.py \
--n_train 500 \
--n_val 120
```

---

## 2. Generate Demo Parking Lot

```bash
python src/generate_demo_lot.py \
--rows 3 \
--cols 6
```

---

## 3. Train Model

```bash
python src/train.py \
--epochs 8 \
--data_root data/dataset \
--out_dir models
```

The trained model will be saved at:

```text
models/best_model.pt
```

---

## 4. Run Parking Detection

```bash
python src/infer.py \
--image data/raw/lot_camera_1.jpg \
--spots data/raw/spots.json \
--model models/best_model.pt \
--out_json app/static/status.json \
--out_image app/static/annotated.jpg
```

---

## 5. Start Dashboard

```bash
cd app

python app.py
```

Open your browser:

```text
http://localhost:5000
```

---

# 🧠 Machine Learning Approach

The system uses **ResNet-18 transfer learning** for parking space classification.

Workflow:

1. Capture parking lot image.
2. Load predefined parking space locations.
3. Crop individual parking spaces.
4. Pass crops through the CNN model.
5. Classify each space:

   * Empty
   * Occupied
6. Display results on the dashboard.

---

# 📊 Dashboard

The dashboard provides:

* Total parking spaces
* Available spaces
* Occupied spaces
* Occupancy percentage
* Color-coded parking visualization
* Prediction confidence
* Annotated parking image

---

# 📚 Dataset

The project supports both synthetic and real parking datasets.

Recommended datasets:

* PKLot
* CNRPark-EXT

Dataset format:

```text
dataset/

├── train/
│   ├── empty/
│   └── occupied/

└── val/
    ├── empty/
    └── occupied/
```

---

# 🚀 Future Improvements

* Real-time CCTV video processing
* YOLO-based vehicle detection
* Automatic parking spot detection
* License plate recognition
* Cloud deployment
* Mobile application support
* Parking availability prediction

---

# 👥 Team Project

Developed as a college group project.

### My Contributions

* Developed Python backend logic
* Integrated deep learning model
* Implemented inference pipeline
* Built Flask APIs
* Connected ML predictions with dashboard

---

## ⭐ If you find this project useful, consider giving it a star!
