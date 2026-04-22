import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
@st.cache_resource
def load_model():
    return YOLO("yolo26s_best_weights.pt")

model = load_model()
st.title("Cricket Ground Detection")
st.write("Upload an image to detect cricket ground")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image)

    results = model.predict(img_array, conf=0.6)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(areas)]

            img = results[0].orig_img.copy()
            x1, y1, x2, y2 = map(int, best_box)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            st.image(img, caption="Best Detection")
    else:
        st.warning("No cricket ground detected in this image.")
        st.image(image, caption="Original Image")
