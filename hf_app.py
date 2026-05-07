import os

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cifar10_model.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Make sure cifar10_model.keras is committed to the Space."
    )

model = load_model(MODEL_PATH)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert input image to model format: (1, 32, 32, 3), normalized to [0, 1]."""
    image = img.convert("RGB").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image):
    if image is None:
        return "Please upload an image."

    processed = preprocess_image(image)
    probs = model.predict(processed, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    top_label = CLASS_NAMES[pred_idx]

    return top_label, scores


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=[
        gr.Label(label="Predicted class"),
        gr.Label(label="Class probabilities"),
    ],
    title="Image Classifier",
    description="Upload an image and the model will predict one of the CIFAR-10 classes.",
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
