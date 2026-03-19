# Sanil-Samir-Mhatre-Knee-GAP-YOLOv11-Based-Knee-Joint-Gap-Detection-and-mJSW-Measurement
# Knee-GAP: YOLOv11-Based Knee Joint Gap Detection and mJSW Measurement

### 1. Overview
This project builds a complete pipeline for **automatic knee joint gap analysis** on X‑ray images using **YOLOv11** and classical image processing. It localizes the knee joint space and measures the **Minimum Joint Space Width (mJSW)** in pixels, providing both batch analysis and an interactive upload tool for new images.

### 2. Main Features
- **YOLOv11 joint-gap detector**: Trained on a knee joint dataset to detect the joint space region with high mAP (~0.995).
- **Automatic mJSW measurement**: Applies CLAHE enhancement and Gaussian smoothing on the YOLO crop, using Canny edge detection to identify femoral and tibial cortical margins.
- **Validation & statistics**: Processes the full validation set (99 images) to generate summary statistics (mean, std dev, min, max) and distribution plots.
- **Interactive upload tool**: A built-in widget allowing users to upload any knee X-ray for real-time gap detection and mJSW measurement.

### 3. Pipeline
1. **Detection**: Load trained YOLOv11 weights and detect the knee joint gap box.
2. **Crop & Preprocess**: Crop to the YOLO box, apply CLAHE and smoothing to improve edge clarity.
3. **Edge‑based Measurement**: Run Canny edge detection, locate boundaries, and compute the shortest vertical gap (mJSW) in pixels.
4. **Batch Validation**: Iterate over the validation set and summarize results.
5. **Interactive Demo**: Real-time inference via a simple upload interface.

### 4. Relation to Existing Work
This approach follows recent deep learning mJSW research showing that automatic measurement can match or outperform manual methods, using a lightweight detection-plus-edges approach instead of heavy segmentation networks.
