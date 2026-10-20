import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

# 1. Title
st.title("Face Mask Detection")
st.write("Upload an image to test")

# 2. Load your saved model file directly
model = load_model("model.h5")

# 3. File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  # Open and show image
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image", use_container_width=True)

  # Predict button
  if st.button("Predict"):
    # Convert PIL image to numpy array
    img = np.array(image)

    # Resize (same as your notebook)
    img = cv2.resize(img, (224, 224))

    # Normalize (same as X / 255)
    img = img / 255.0

    # Expand dimensions to add batch size: (1, 224, 224, 3)
    img_input = np.expand_dims(img, axis=0)

    # Predict using your model.h5
    prediction = model.predict(img_input)[0][0]

    # Output result
    if prediction > 0.5:
      st.success("Result: With Mask 😷")
    else:
      st.error("Result: Without Mask ❌")