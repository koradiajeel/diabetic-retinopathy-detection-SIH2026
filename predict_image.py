import tensorflow as tf
from archive.train_retinopathy_v2 import predict_single_image

model = tf.keras.models.load_model(
    r"D:\SIH\retinopathy_output_v2\final_retinopathy_model.keras"
)

image_path = r"C:\Users\Korad\OneDrive\Pictures\Screenshots\Screenshot 2026-09-09 170224.png"

grade, label, confidence = predict_single_image(model, image_path)

print(f"\nGrade: {grade}")
print(f"Result: {label}")
print(f"Confidence: {confidence:.1f}%")