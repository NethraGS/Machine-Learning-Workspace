from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI(title="CIFAR-10 Image Classification API")

# Load CIFAR-10 model
model = tf.keras.models.load_model("cifar_model.h5")

# CIFAR-10 class labels (ORDER MATTERS)
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

@app.get("/")
def home():
    return {"message": "CIFAR-10 Prediction API is running 🚀"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()

    # Convert bytes to OpenCV image (COLOR!)
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # Resize to CIFAR-10 size
    img = cv2.resize(img, (32, 32))

    # Convert BGR → RGB (IMPORTANT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = img.reshape(1, 32, 32, 3)

    # Predict
    prediction = model.predict(img)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return {
        "predicted_class": predicted_class,
        "label": class_names[predicted_class],
        "confidence": f"{confidence:.2f}%"
    }
