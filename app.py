import streamlit as st
from keras.layers import TFSMLayer
import numpy as np
from PIL import Image

# Load model wrapper
@st.cache_resource
def load_model():
    # 'model' is the folder path to the extracted SavedModel
    return TFSMLayer("model", call_endpoint="serving_default")

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

labels = ["Good", "Defective"]

st.title("Defect Detection with Keras 3")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)
    preds = model(input_data)  # TFSMLayer returns a TensorFlow tensor
    pred_tensor = list(preds.values())[0]  # get first tensor from dict
    preds_np = pred_tensor.numpy()         # convert tensor to numpy array

    pred_label = labels[np.argmax(preds_np)]
    confidence = np.max(preds_np)

    st.write(f"Prediction: **{pred_label}**")
    st.write(f"Confidence: {confidence:.2f}")
