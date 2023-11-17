import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the model (update with your model's path)
@st.cache_data
def load_model():
    return tf.keras.models.load_model('MobileNetV2.h5')

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((128, 128))  # Resize (adjust according to your model)
    img_array = np.array(img) / 255.0  # Normalize (if your model requires)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Class names mapping (replace with your actual class names)
class_names = {0: "Parasitized", 1: "Uninfected", 2: "Unknown Class"}  # Example

# Function to display predictions
def display_predictions(predictions):
    # Assuming the predictions are class probabilities
    class_index = np.argmax(predictions, axis=1)[0]
    class_probability = np.max(predictions, axis=1)[0]
    class_name = class_names.get(class_index, "Unknown Class")

    st.write(f"Prediction: {class_name} with probability {class_probability * 100:.2f}")

# Load the model
model = load_model()

# Streamlit UI
st.title("Malaria cell Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image.squeeze(), caption='Uploaded Image', use_column_width=True)
	# Predict button
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            predictions = model.predict(image)
            display_predictions(predictions)
