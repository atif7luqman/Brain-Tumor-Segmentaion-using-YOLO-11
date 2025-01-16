import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load the YOLO model
MODEL_PATH = "best.pt"  # Update with your model's path
model = YOLO(MODEL_PATH)

# Streamlit app configuration
st.set_page_config(page_title="YOLO Segmentation", layout="centered")

# Streamlit UI
st.title("YOLO Segmentation Model Deployment")
st.write("Upload an image to perform segmentation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")

    # Convert image to OpenCV format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform segmentation
    results = model(image_bgr)

    # Draw results on the image
    annotated_image = results[0].plot()

    # Convert BGR to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the results
    st.image(annotated_image_rgb, caption="Segmented Image", use_container_width=True)

    # Optional: Display additional information
    st.write("Segmentation Results:")
    for box in results[0].boxes:
        st.write(f"Class: {box.cls}, Confidence: {float(box.conf):.2f}")

    
