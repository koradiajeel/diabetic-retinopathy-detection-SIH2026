from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image


app = FastAPI(title="Diabetic Retinopathy Detection API")


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "final_retinopathy_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# MODEL SETTINGS
# ============================================================

IMAGE_SIZE = 300

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# ============================================================
# PREPROCESSING
# Same logic as train_retinopathy_v2.py
# ============================================================

def crop_black_border(image: tf.Tensor) -> tf.Tensor:
    """Crop empty black margins so the retina occupies more pixels."""

    brightness = tf.reduce_max(image, axis=-1)

    mask = brightness > 10

    rows = tf.cast(
        tf.where(tf.reduce_any(mask, axis=1))[:, 0],
        tf.int32
    )

    columns = tf.cast(
        tf.where(tf.reduce_any(mask, axis=0))[:, 0],
        tf.int32
    )

    def crop():
        top = rows[0]
        bottom = rows[-1]

        left = columns[0]
        right = columns[-1]

        return tf.image.crop_to_bounding_box(
            image,
            top,
            left,
            bottom - top + 1,
            right - left + 1
        )

    has_content = tf.logical_and(
        tf.size(rows) > 0,
        tf.size(columns) > 0
    )

    return tf.cond(
        has_content,
        crop,
        lambda: image
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Diabetic Retinopathy API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        image_bytes = await file.read()

        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])


        # ----------------------------------------------------
        # SAME preprocessing as train_retinopathy_v2.py
        # ----------------------------------------------------

        image = crop_black_border(image)

        image = tf.image.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True
        )

        # IMPORTANT:
        # Do NOT divide by 255.
        # Original model uses 0-255 float values.

        image = tf.cast(
            image,
            tf.float32
        )


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        scores = model.predict(
            tf.expand_dims(image, 0),
            verbose=0
        )[0]


        # ----------------------------------------------------
        # Get prediction
        # ----------------------------------------------------

        predicted_class = int(
            np.argmax(scores)
        )

        predicted_label = CLASS_NAMES[
            predicted_class
        ]

        confidence = float(
            scores[predicted_class] * 100
        )


        # ----------------------------------------------------
        # All probabilities
        # ----------------------------------------------------

        probabilities = {
            CLASS_NAMES[i]: round(
                float(scores[i]) * 100,
                2
            )
            for i in range(len(CLASS_NAMES))
        }


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "filename": file.filename,

            "predicted_class": predicted_class,

            "predicted_label": predicted_label,

            "confidence": round(
                confidence,
                2
            ),

            "probabilities": probabilities
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )