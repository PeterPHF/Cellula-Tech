import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

DATA_DIR_PATH = os.path.join(os.getcwd(), 'Teeth_Dataset')
TEST_DATA_PATH = os.path.join(DATA_DIR_PATH, 'Testing')
classes = os.listdir(TEST_DATA_PATH)
# Load your model
model = tf.keras.models.load_model('VGG16_fine_tuned.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((120, 120))  # Resize to the size expected by the model
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app interface
st.title('Deep Learning Model Deployment with Streamlit')
st.write('Upload an image to classify')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write('')

    # Predict and display the result
    if st.button('Predict'):
        prediction = predict(image)
        idx = np.argmax(prediction)
        st.write(f'Prediction: {classes[idx]}')  # Modify based on your model's output
