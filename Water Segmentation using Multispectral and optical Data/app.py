from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import tifffile as tiff

# Initialize Flask app
app = Flask(__name__)

# Load your DeepLabV3+ model
model = tf.keras.models.load_model('pre_trained_deeplabv3_plus_with_ES_MC.keras')

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def preprocess_image(img_path):
    img = tiff.imread(img_path)
    img = np.expand_dims(img, axis=0)
    return img

# Prediction and post-processing function
def predict_mask(image_path):
    
    img = preprocess_image(image_path)
    pred_mask = model.predict(img)[0]
    pred_mask[pred_mask >= 0.5] = 1
    pred_mask[pred_mask < 0.5] = 0
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask = tf.squeeze(pred_mask).numpy()
    return pred_mask

# Route for the home page
@app.route('/')
def index():
    return render_template('upload.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run prediction
        predicted_mask = predict_mask(filepath)

        # Convert mask to an image and save it
        # Save the predicted mask as an image
        # Save the predicted mask as an image
        mask_img = Image.fromarray((predicted_mask[:,:, 0] * 255).astype(np.uint8))  # Scale to 0-255
        mask_filename = "mask.png"
        mask_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
        mask_img.save(mask_filepath)

        # Serve the mask image file back to the user
        return send_file(mask_filepath, mimetype='image/png')


    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
