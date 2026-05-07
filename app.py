import os
from io import BytesIO

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Define CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cifar10_model.keras')
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print(f"Error: {MODEL_PATH} not found. Please make sure you saved the model from the notebook.")
    model = None

def preprocess_image(img_path):
    """
    Preprocess the image to match the model's input shape (32, 32, 3)
    and normalize pixel values to [0, 1].
    """
    img = Image.open(img_path).convert('RGB').resize((32, 32))
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize as done in training
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the uploaded image in memory to avoid filesystem issues on deployment.
        processed_img = preprocess_image(BytesIO(file.read()))
        prediction = model.predict(processed_img)

        # Get the class with the highest probability
        predicted_class_idx = np.argmax(prediction)
        predicted_label = CLASS_NAMES[predicted_class_idx]

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
