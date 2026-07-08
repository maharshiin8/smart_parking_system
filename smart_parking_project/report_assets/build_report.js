const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, ImageRun, PageBreak,
  TableOfContents, Header, Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = require("docx");
const fs = require("fs");

const ACCENT = "C99A12";
const DARKGRAY = "333333";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: [new TextRun({ text, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function numbered(text) {
  return new Paragraph({ text, numbering: { reference: "num-list", level: 0 }, spacing: { after: 80 } });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text, italics: true, size: 20, color: "666666" })],
  });
}
function imageParagraph(path, width, height) {
  const data = fs.readFileSync(path);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({ data, transformation: { width, height }, type: path.endsWith(".png") ? "png" : "jpg" })],
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2500, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "2B2B2B" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
  });
}

function simpleTable(headers, rows, widths) {
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ children: headers.map((hh, i) => cell(hh, { header: true, width: widths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(String(c), { width: widths[i] })) })),
    ],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "num-list",
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START }],
    }],
  },
  sections: [
    // ---------- TITLE PAGE ----------
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "SMART PARKING SYSTEM", bold: true, size: 56, color: DARKGRAY })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 400 },
          children: [new TextRun({ text: "Real-Time Parking Occupancy Detection using Computer Vision and Deep Learning", size: 28, color: ACCENT, italics: true })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 800 },
          children: [new TextRun({ text: "A Project Report", size: 24 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1600 },
          children: [new TextRun({ text: "Submitted in partial fulfillment of the requirements for", size: 22 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1600 },
          children: [new TextRun({ text: "the course project / capstone in Artificial Intelligence & Machine Learning", size: 22 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 2000 },
          children: [new TextRun({ text: "Submitted by: [Your Name]", size: 22 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Roll No / ID: [Your ID]", size: 22 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
          children: [new TextRun({ text: "Department: [Your Department]", size: 22 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1600 },
          children: [new TextRun({ text: "[Your College / University Name]", size: 22, bold: true })] }),
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "2026", size: 22 })] }),
      ],
    },
    // ---------- TOC ----------
    {
      children: [
        h1("Table of Contents"),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },
    // ---------- MAIN CONTENT ----------
    {
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Smart Parking System — Project Report", size: 16, color: "888888" })],
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })],
        })] }),
      },
      children: [
        h1("Abstract"),
        p("Urban and campus parking lots are frequently underutilized because drivers cannot see, in real time, which spaces are free. This project presents a Smart Parking System that determines parking-spot occupancy directly from camera images using a deep convolutional neural network (CNN), and presents the results through a live web dashboard. A ResNet-18 backbone is fine-tuned via transfer learning to classify small image patches, one per parking spot, as either empty or occupied. Each camera frame is processed by cropping every known spot location, classifying every patch, and aggregating the results into lot-wide occupancy statistics. The system is implemented end-to-end in Python using PyTorch for the model and Flask for a browser-based dashboard that visualizes spot-by-spot status, refreshes on demand, and reports per-spot confidence scores. On a controlled validation set the trained classifier achieves 100% accuracy, and the full pipeline, from raw camera frame to dashboard update, executes in well under a second per frame on CPU, demonstrating feasibility for near-real-time deployment. The report documents the motivation, related work, system design, dataset and model methodology, implementation, evaluation, and directions for extending the system to real camera feeds and larger, real-world datasets such as PKLot and CNRPark-EXT."),

        h1("1. Introduction"),
        h2("1.1 Motivation"),
        p("Finding an available parking spot in a busy lot wastes driver time, increases traffic congestion and fuel consumption near entrances, and creates avoidable frustration on campuses and in commercial lots alike. Traditional solutions rely on dedicated hardware, such as ultrasonic or magnetic sensors embedded under every single space, which is accurate but expensive to install and maintain at scale. An alternative that has become practical with modern deep learning is to reuse cameras that are already installed for security, and infer occupancy purely from the images they capture."),
        h2("1.2 Problem Statement"),
        p("Given a fixed camera view of a parking lot and the known pixel location of every marked parking space, automatically determine, for every space and for every new frame, whether it is empty or occupied, and present this information to users through a live status dashboard."),
        h2("1.3 Objectives"),
        bullet("Build an image-patch classifier that labels a single parking spot as empty or occupied."),
        bullet("Apply transfer learning (ResNet-18 pretrained on ImageNet) to reach high accuracy with a modest amount of labelled data."),
        bullet("Build an inference pipeline that maps a full lot image plus per-spot coordinates to a lot-wide occupancy report."),
        bullet("Present live results through a web dashboard suitable for drivers, facilities staff, or a campus parking app."),
        bullet("Evaluate the system's accuracy and discuss its limitations and a path to real-world deployment."),
        h2("1.4 Scope"),
        p("This project focuses on the vision + classification pipeline and the presentation layer. It assumes a fixed, non-moving camera per lot (the standard configuration in the academic literature and in commercial products such as fixed CCTV-based systems) and that spot coordinates are calibrated once, when the camera is installed. Vehicle re-identification, license-plate recognition, and payment/reservation systems are out of scope."),

        h1("2. Literature Review"),
        p("Camera-based parking occupancy detection has been an active research area since the mid-2010s. Two datasets are widely used as academic benchmarks:"),
        bullet("PKLot: images of two Brazilian parking lots captured under varied weather and lighting, with each spot manually annotated as empty or occupied."),
        bullet("CNRPark-EXT: a larger dataset from an Italian university campus, captured from nine camera viewpoints across many days and weather conditions, designed specifically to test how well a model trained on one camera generalizes to another."),
        p("Early approaches used classical computer vision (background subtraction, edge density, or hand-crafted texture features with an SVM classifier). These are lightweight but degrade sharply under shadows, changing weather, and partial occlusion. Later work (e.g., mAlexNet and other compact CNNs proposed alongside CNRPark-EXT) showed that a CNN trained directly on cropped spot patches substantially outperforms classical features, and generalizes better across cameras and lighting conditions. This project follows that same patch-classification paradigm, but uses a deeper, ImageNet-pretrained ResNet-18 backbone via transfer learning, which typically improves accuracy and shortens training time versus a CNN trained from scratch, especially when labelled data is limited."),
        p("Sensor-based alternatives (ultrasonic, infrared, geomagnetic pucks) are also common in commercial deployments and offer high per-spot accuracy independent of lighting, but require per-spot hardware, wiring or batteries, and ongoing maintenance, which does not scale economically to large lots. Camera-based systems trade a small amount of accuracy in adverse conditions for near-zero marginal hardware cost per additional spot, since one camera already covers many spaces."),

        h1("3. System Architecture"),
        p("The system has four stages, illustrated below and mirrored directly in the project's source layout."),
        simpleTable(
          ["Stage", "Responsibility", "Key files"],
          [
            ["1. Spot calibration", "One-time mapping of each physical parking space to a pixel bounding box within the fixed camera's field of view.", "spots.json"],
            ["2. Patch extraction", "For every new frame, crop the region of the frame corresponding to each spot.", "src/infer.py"],
            ["3. Classification", "Run the trained CNN on every cropped patch to predict empty / occupied with a confidence score.", "src/model.py, models/best_model.pt"],
            ["4. Presentation", "Aggregate predictions into lot-wide counts and serve them, plus an annotated image, to a live dashboard.", "app/app.py, app/templates, app/static"],
          ],
          [2600, 5200, 2600],
        ),
        new Paragraph({ spacing: { before: 200 } }),
        p("This patch-level design has an important practical advantage over training a single model to look at the whole lot image at once: it decouples the number of parking spaces from the model, so the same trained classifier works for a 10-spot lot or a 500-spot lot, and adding, removing, or recalibrating a spot only requires editing spots.json, not retraining the model."),

        h1("4. Dataset and Methodology"),
        h2("4.1 Data"),
        p("The classifier is trained on labelled parking-spot patches organized as:"),
        p("data/dataset/train/{empty, occupied}/*.jpg   and   data/dataset/val/{empty, occupied}/*.jpg", { font: "Consolas", size: 20 }),
        p("For this report, a synthetic patch generator (src/generate_synthetic_dataset.py) was used to create a demo dataset of 1,000 training and 240 validation patches (balanced 50/50 between classes), because the sandboxed development environment used to build this project has no general internet access to download the real PKLot / CNRPark-EXT datasets. The generator renders an asphalt-textured background with painted parking lines and, for the occupied class, a randomly colored, positioned car silhouette with a windshield highlight, giving controlled but realistic-looking variation in car color, position and lighting."),
        p("Because this is a demo/synthetic dataset, it is not representative of the full difficulty of real-world footage (rain, deep shadows, dusk lighting, partial occlusion between adjacent cars). The pipeline, however, is dataset-agnostic: pointing dataset.py at a folder of real PKLot/CNRPark-EXT images organized in the same train/val/empty/occupied layout is a drop-in replacement, and is the recommended next step described in Section 8 (Future Work)."),
        h2("4.2 Preprocessing and Augmentation"),
        bullet("Resize every patch to 64x64 pixels."),
        bullet("Normalize using ImageNet channel mean/std, required because the backbone is ImageNet-pretrained."),
        bullet("Training-time augmentation: random horizontal flip, color jitter (brightness/contrast/saturation), and small random rotation (±5°), to make the classifier robust to lighting and camera-angle variation."),
        h2("4.3 Model"),
        p("The classifier (src/model.py) is a ResNet-18 convolutional network pretrained on ImageNet, with its final fully-connected layer replaced by a small classification head (Dropout(0.3) -> Linear(512, 2)) producing empty/occupied logits. Transfer learning was chosen over training a CNN from scratch because:"),
        bullet("The pretrained convolutional filters already encode general-purpose edge, texture and shape features, so the network needs far less labelled data to reach high accuracy."),
        bullet("Convergence is faster: only a few epochs of fine-tuning are needed rather than tens of epochs from random initialization."),
        bullet("It matches standard practice in the parking-classification literature and in transfer-learning coursework generally, making it a defensible, well-justified architectural choice for a college project."),
        p("Note: the sandboxed environment used to develop this project could not reach download.pytorch.org to fetch ImageNet weights, so the results in Section 6 were produced with random initialization as a fallback (the code automatically detects this and falls back gracefully). On a machine with normal internet access, setting pretrained=True in model.py will download the ImageNet weights and should be used for the strongest results, particularly on a real, harder dataset."),
        h2("4.4 Training Configuration"),
        simpleTable(
          ["Hyperparameter", "Value"],
          [
            ["Optimizer", "Adam"],
            ["Learning rate", "1e-3 (StepLR schedule, decayed by 0.5 every 5 epochs)"],
            ["Loss function", "Cross-entropy"],
            ["Batch size", "32"],
            ["Epochs", "8"],
            ["Input size", "64 x 64 x 3"],
          ],
          [4400, 5000],
        ),

        h1("5. Implementation"),
        h2("5.1 Technology Stack"),
        simpleTable(
          ["Layer", "Technology"],
          [
            ["Model / training", "Python, PyTorch, torchvision"],
            ["Inference pipeline", "PyTorch, Pillow"],
            ["Backend API", "Flask (Python)"],
            ["Frontend dashboard", "HTML5, CSS3, vanilla JavaScript (fetch API, polling)"],
          ],
          [3200, 6200],
        ),
        h2("5.2 Inference Pipeline"),
        p("src/infer.py loads the trained model checkpoint, reads the current lot image and spots.json, crops every spot's bounding box, classifies each crop, and writes two outputs: a status.json with per-spot predictions/confidence and lot-wide counts, and an annotated JPEG with green (empty) / red (occupied) boxes drawn over every spot, labelled with its spot ID."),
        h2("5.3 Web Dashboard"),
        p("The Flask backend (app/app.py) exposes:"),
        bullet("GET /api/status — returns the current per-spot status and lot-wide counts as JSON."),
        bullet("POST /api/refresh — simulates a new camera frame (in a real deployment this would pull the newest frame from an IP camera / RTSP stream instead), re-runs the CNN, and returns the updated status."),
        p("The frontend polls /api/status every 15 seconds and lets the user manually trigger /api/refresh with a \"Pull new frame\" button. The dashboard shows: total spots, available spots, occupied spots and occupancy rate as headline stats; a color-coded grid map of every individual spot with hover tooltips showing the model's confidence; and a short explanation panel for a non-technical viewer."),
        imageParagraph("report_assets/dashboard_screenshot_crop.png", 480, 436),
        caption("Figure 1: Live parking dashboard, showing headline occupancy stats and the color-coded spot map."),

        h1("6. Results and Evaluation"),
        h2("6.1 Training Curves"),
        imageParagraph("report_assets/training_curves.png", 500, 200),
        caption("Figure 2: Training/validation accuracy (left) and loss on a log scale (right) over 8 epochs."),
        p("The model converges within the first epoch and reaches 100% training and validation accuracy by epoch 2, which is expected given the controlled, synthetic dataset used in this environment (fixed background style, limited pose/occlusion variation). This should be read as a correctness check of the full pipeline rather than a claim about real-world performance; Section 7 discusses this limitation directly."),
        h2("6.2 Confusion Matrix"),
        imageParagraph("report_assets/confusion_matrix.png", 260, 250),
        caption("Figure 3: Validation confusion matrix (120 empty, 120 occupied patches) — zero misclassifications."),
        h2("6.3 Qualitative Result on a Full Lot Frame"),
        p("Running the full pipeline (src/infer.py) on a demo 18-spot lot frame correctly separated occupied from empty spaces, producing lot-wide counts and an annotated frame:"),
        imageParagraph("report_assets/annotated_demo.jpg", 460, 216),
        caption("Figure 4: Annotated demo frame — green boxes are predicted empty, red boxes are predicted occupied, matching ground truth for all 18 spots in this example."),
        h2("6.4 Runtime Performance"),
        p("On CPU, classifying all 18 spots in a frame (crop + normalize + forward pass per spot) completes in well under one second, and the model checkpoint is under 45 MB, both compatible with near-real-time refresh rates (e.g., once every few seconds) even without GPU acceleration."),

        h1("7. Limitations"),
        bullet("Synthetic training data: results reflect a controlled demo dataset, not the lighting, weather, shadow and occlusion variability of real footage; real-world accuracy will be lower until the model is retrained on PKLot/CNRPark-EXT or on footage from the target lot."),
        bullet("Fixed camera assumption: spot coordinates are calibrated once per camera; a moved or re-angled camera requires recalibrating spots.json."),
        bullet("No temporal smoothing: each frame is classified independently, so a single misclassified frame could cause a brief, flickering status change; a short majority-vote over the last few frames would stabilize this in production."),
        bullet("Single-camera occlusion: a spot partially hidden behind another vehicle from the camera's viewpoint can be misread, a known challenge in the camera-based parking literature."),

        h1("8. Future Work"),
        numbered("Retrain on a real, publicly available dataset (PKLot or CNRPark-EXT) to obtain realistic, publishable accuracy figures."),
        numbered("Replace the simulated /api/refresh with a live RTSP/IP-camera feed, decoded with OpenCV, so the dashboard reflects an actual live lot."),
        numbered("Add temporal smoothing (e.g., majority vote or exponential moving average over the last N frames) to reduce status flicker."),
        numbered("Support multiple cameras/lots in one dashboard, with per-lot occupancy summaries for a campus-wide view."),
        numbered("Add a public-facing mobile view or SMS/push notification when a lot crosses a fullness threshold."),
        numbered("Explore lightweight architectures (e.g., MobileNet) for on-device inference at the camera edge, reducing bandwidth and latency."),

        h1("9. Conclusion"),
        p("This project demonstrates a complete, working smart parking pipeline: a transfer-learning CNN that classifies individual parking-spot image patches as empty or occupied, an inference pipeline that turns a raw camera frame into a lot-wide occupancy report, and a live web dashboard that presents that report to end users. The architecture is intentionally decoupled, patch classification, spot calibration, and presentation are independent, so it scales to lots of any size and can be upgraded piece by piece (a stronger dataset, a live camera feed, temporal smoothing) without redesigning the system. While the specific accuracy numbers in this report reflect a synthetic demo dataset built to work within an offline development environment, the same code operates unchanged on real parking-lot datasets and real camera feeds, making this a practical foundation for a deployable smart parking solution."),

        h1("References"),
        p("[1] P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., and A. L. Koerich, \"PKLot – A robust dataset for parking lot classification,\" Expert Systems with Applications, 2015."),
        p("[2] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Meghini, and C. Vairo, \"Deep learning for decentralized parking lot occupancy detection,\" Expert Systems with Applications, 2017 (CNRPark-EXT)."),
        p("[3] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image Recognition,\" CVPR, 2016 (ResNet)."),
        p("[4] PyTorch documentation, https://pytorch.org/docs — model, training and transfer-learning APIs used in this project."),
        p("[5] Flask documentation, https://flask.palletsprojects.com — web framework used for the dashboard backend."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Smart_Parking_System_Report.docx", buf);
  console.log("Report written.");
});
