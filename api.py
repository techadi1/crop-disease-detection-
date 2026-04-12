from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("apple_leaf_model.keras")

# Full PlantVillage 38-class list (used if the model outputs 38 logits)
class_names_38 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# The 4 specific apple classes the frontend expects in its graph
apple_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))

    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, 0)

    # Model outputs logits or probabilities for N classes
    preds = model.predict(arr, verbose=0)[0]
    num_classes = preds.shape[0]

    # Dynamically pick class names based on model shape
    if num_classes == 4:
        # Apple-only model (4-way classifier)
        class_names = apple_classes
    elif num_classes == 38:
        # Full PlantVillage model
        class_names = class_names_38
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Model output has unexpected number of classes: {num_classes}.",
        )

    max_prob = float(np.max(preds))
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]

    # --- Smart Confidence + Entropy Check ---
    entropy = -float(np.sum(preds * np.log(preds + 1e-12)))
    MAX_PROB_THRESHOLD = 0.50   # require at least 50% probability to accept
    ENTROPY_THRESHOLD = 1.5     # if entropy > 1.5 (nats) -> uncertain, reject

    # For a 38-class model, also reject if it's confidently predicting a non-apple plant.
    reject_non_apple = num_classes == 38 and "Apple" not in pred_class

    low_confidence = max_prob < MAX_PROB_THRESHOLD or entropy > ENTROPY_THRESHOLD

    warning = None
    if low_confidence or reject_non_apple:
        warning = "Low confidence prediction or possibly not an apple leaf"

    # Build 4-class apple probability breakdown for the frontend graph
    probabilities = []
    for app_class in apple_classes:
        if app_class in class_names:
            idx = class_names.index(app_class)
            value = float(preds[idx]) * 100
        else:
            # If the underlying model is 4-class apple-only, this still works;
            # if it's a 38-class model, non-apple classes are simply omitted.
            value = 0.0

        probabilities.append(
            {
                "name": app_class.replace("Apple___", "").replace("_", " "),
                "value": value,
            }
        )

    return {
        "prediction": pred_class,
        "confidence": max_prob * 100,
        "probabilities": probabilities,
        "warning": warning
    }