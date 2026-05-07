import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Define CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
MODEL_PATH = 'cifar10_model.keras'
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
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
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
        # Save uploaded file temporarily
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)

        # Preprocess and predict
        processed_img = preprocess_image(temp_path)
        prediction = model.predict(processed_img)

        # Get the class with the highest probability
        predicted_class_idx = np.argmax(prediction)
        predicted_label = CLASS_NAMES[predicted_class_idx]

        # Cleanup
        os.remove(temp_path)

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
