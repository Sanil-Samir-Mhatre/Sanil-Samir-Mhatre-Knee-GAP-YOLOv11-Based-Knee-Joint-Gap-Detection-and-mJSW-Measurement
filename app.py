import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. Load Model ---
@st.cache_resource
def load_model():
    return YOLO('Knee-GAP-YOLOv11-Based-Knee-Joint-Gap-Detection-and-mJSW-Measurement/yolo11n.pt')

model = load_model()

# --- 2. Helper Functions (From our Colab) ---
def get_medical_metrics(pixel_width, mm_per_pixel=0.15):
    mm_width = pixel_width * mm_per_pixel
    if mm_width >= 3.0:
        status = 'Rough Estimation: Normal / Healthy Range'
    elif 2.0 <= mm_width < 3.0:
        status = 'Rough Estimation: Possible Mild Narrowing (Screening Required)'
    else:
        status = 'Rough Estimation: Possible Significant Narrowing (Further Clinical Evaluation Advised)'
    return mm_width, status

def calculate_mjsw_robust(img, model):
    results = model(img)[0]
    if len(results.boxes) == 0: return None, 'No detection'
    box = results.boxes.xyxy[0].cpu().numpy().astype(int)
    crop = img[box[1]:box[3], box[0]:box[2]]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    h, w = edges.shape
    min_dist = float('inf')
    best_coords = None
    for x in range(int(w * 0.25), int(w * 0.75)):
        y_indices = np.where(edges[:, x] > 0)[0]
        if len(y_indices) >= 2:
            dist = y_indices[-1] - y_indices[0]
            if 2 < dist < min_dist:
                min_dist, best_coords = dist, (x, y_indices[0], y_indices[-1])
    if best_coords:
        x, y1, y2 = best_coords
        cv2.line(crop, (x, y1), (x, y2), (0, 255, 0), 2)
        return crop, min_dist
    return crop, 'Measurement failed'

# --- 3. Streamlit UI ---
st.title('Knee Joint Gap Analysis')
st.write('Upload a Knee X-ray for automated mJSW measurement.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('Analyzing...'):
        viz, mjsw = calculate_mjsw_robust(image, model)
    
    if isinstance(mjsw, (int, float, np.integer)):
        mm_val, status = get_medical_metrics(mjsw)
        st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption=f'mJSW: {mm_val:.2f}mm')
        st.success(f'**Result:** {status}')
        st.info(f'**Raw Measurement:** {mjsw} pixels')
    else:
        st.error(f'Error: {mjsw}')