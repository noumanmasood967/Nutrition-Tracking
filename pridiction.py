import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = "D:/New folder/nouman/niutition/train/best_food_model.keras"

CLASSES_SOURCE_DIR = "D:/New folder/nouman/niutition/Valid"

TARGET_IMAGE_SIZE = (224, 224)

IMAGE_TO_PREDICT_PATH = "D:/New folder/nouman/niutition/Train/Fries/Fries-Train (19).jpeg"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure the MODEL_PATH is correct and the model file exists.")
    exit()

try:
    class_names = sorted([d for d in os.listdir(CLASSES_SOURCE_DIR) if os.path.isdir(os.path.join(CLASSES_SOURCE_DIR, d))])
    if '.DS_Store' in class_names:
        class_names.remove('.DS_Store')
    print(f"✅ Detected {len(class_names)} classes: {class_names}")
except Exception as e:
    print(f"❌ Error getting class names from {CLASSES_SOURCE_DIR}: {e}")
    print("Please ensure CLASSES_SOURCE_DIR points to a directory with subfolders for each class.")
    exit()

try:
    img = image.load_img(IMAGE_TO_PREDICT_PATH, target_size=TARGET_IMAGE_SIZE)

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array /= 255.0

    print(f"\n✅ Image '{os.path.basename(IMAGE_TO_PREDICT_PATH)}' loaded and preprocessed.")
except FileNotFoundError:
    print(f"❌ Error: Image file not found at {IMAGE_TO_PREDICT_PATH}")
    exit()
except Exception as e:
    print(f"❌ Error during image loading or preprocessing: {e}")
    exit()

print("🚀 Making prediction...")
predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions[0])

predicted_class_name = class_names[predicted_class_index]

confidence = predictions[0][predicted_class_index] * 100

print("\n--- Prediction Results ---")
print(f"Input Image: {os.path.basename(IMAGE_TO_PREDICT_PATH)}")
print(f"Predicted Class: '{predicted_class_name}'")
print(f"Confidence: {confidence:.2f}%")

print("\nDetailed Probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"  - {class_names[i]}: {prob * 100:.2f}%")

if confidence > 80:
    print("\n✨ The model is highly confident in its prediction!")
elif confidence > 50:
    print("\n👍 The model has made a prediction with moderate confidence.")
else:
    print("\n⚠️ The model's confidence is low. The prediction might not be accurate.")
