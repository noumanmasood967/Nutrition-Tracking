import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS # Used for handling Cross-Origin Resource Sharing
import mysql.connector # For MySQL database interactions

# Import Keras components if needed (already imported, just for clarity)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

# --- Flask App Initialization ---
app = Flask(__name__)
# Configure CORS: Allow requests from any origin (*) to the /food endpoint
# This is crucial for your frontend running on a different port to communicate with this backend.
CORS(app, resources={r"/food": {"origins": "*"}}) 

# --- Model Loading ---
# Define the path to your trained model
MODEL_PATH = "D:/New folder/nouman/niutition/train/best_food_model.keras"

# Global variable to hold the loaded model
model = None 
try:
    print(f"🔄 Loading model from: {MODEL_PATH}")
    # Load the Keras model. This step can take some time.
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    # If model loading fails, print an error and set model to None
    print(f"❌ Error loading model: {e}")
    model = None

# --- MySQL Database Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "", # Your MySQL password (empty if none)
    "database": "food_tracker" # Name of your database
}

def get_db_connection():
    """Establish a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ Database connection established.")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Database connection error: {err}")
        return None

# --- Class Names Loading ---
# Path to your dataset, used to dynamically fetch class names (food categories)
DATASET_PATH = "D:/New folder/nouman/niutition/Train"
if os.path.exists(DATASET_PATH):
    # Dynamically get class names from subdirectories in the dataset path
    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    print("✅ Class Names Loaded:", class_names)
    print(f"Total classes: {len(class_names)}")
else:
    # Fallback to default class names if the dataset path is not found
    class_names = ["Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries",
                   "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"]
    print(f"⚠️ Warning: Dataset path '{DATASET_PATH}' not found. Using Default Class Names.")

# --- Image Preprocessing Function ---
def preprocess_image(image_file):
    """
    Reads, resizes, and normalizes image for prediction with proper color format.
    Handles file seeking to ensure the image can be read multiple times if needed.
    """
    try:
        # Reset file pointer to beginning to ensure full content is read
        image_file.seek(0) 
        
        # Read image data as bytes and decode it using OpenCV
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR) # Loads image as BGR by default
        
        if image is None:
            print("❌ Error: Image decoding failed. Check file format or corruption.")
            return None
        
        print(f"Original image shape: {image.shape}")
        
        # CRITICAL FIX: Convert BGR (OpenCV default) to RGB (TensorFlow/Keras model expectation)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to the input size expected by the model (e.g., 224x224 for MobileNetV2)
        image = cv2.resize(image, (224, 224))
        print(f"Resized image shape: {image.shape}")
        
        # Normalize pixel values from [0, 255] to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add a batch dimension: (height, width, channels) -> (1, height, width, channels)
        image = np.expand_dims(image, axis=0)
        print(f"Final preprocessed shape: {image.shape}")
        
        return image
    except Exception as e:
        print(f"❌ Error during image preprocessing: {e}")
        return None

# --- Food Prediction Function ---
def predict_food_from_image(image_file, actual_label=None):
    """
    Predicts the food category from an image using the loaded CNN model.
    Includes confidence thresholding and fetches nutrition data.
    """
    try:
        if model is None:
            return {"error": "Model not loaded. Cannot make prediction."}

        # Preprocess the uploaded image
        img = preprocess_image(image_file)
        if img is None:
            return {"error": "Invalid or unreadable image format. Please upload a valid image."}

        print("🔄 Making prediction...")
        # Get raw prediction probabilities from the model
        raw_output = model.predict(img, verbose=0)
        
        # --- Debugging: Print all prediction probabilities ---
        print("\n📊 PREDICTION ANALYSIS:")
        print("="*50)
        for i, class_name in enumerate(class_names):
            # Ensure index is within bounds of raw_output
            probability = raw_output[0][i] if i < len(raw_output[0]) else 0
            print(f"{class_name:15}: {probability:.4f} ({probability*100:.2f}%)")
        print("="*50)
        # --- End Debugging ---
        
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(raw_output)
        # Get the confidence score for the predicted class
        confidence = float(raw_output[0][predicted_class_index])
        
        # Validate if the predicted index is within the bounds of known class names
        if predicted_class_index >= len(class_names):
            return {"error": "Prediction class index out of range. Model output does not match expected classes."}

        food_name = class_names[predicted_class_index]
        confidence_percentage = confidence * 100
        
        print(f"✅ Predicted: {food_name} with confidence {confidence_percentage:.2f}%")

        if actual_label:
            print(f"🎯 Actual Label (for comparison): {actual_label}")

        # --- Confidence Thresholding ---
        # If confidence is below 50%, return a low confidence message
        if confidence_percentage < 50:
            print(f"⚠️ LOW CONFIDENCE: {confidence_percentage:.2f}% < 50%. Suggesting a clearer image.")
            return {
                "low_confidence": True,
                "food_name": food_name, # Still return the best guess, but with a warning
                "confidence": f"{confidence_percentage:.2f}%",
                "message": "Low confidence detected. Please upload a clearer image of the food item.",
                "suggestion": "Try taking a photo with better lighting and ensure the food item is clearly visible in the center."
            }

        # --- High Confidence Prediction ---
        print(f"✅ HIGH CONFIDENCE: {confidence_percentage:.2f}% >= 50%. Proceeding with nutrition data.")
        
        # Determine if the prediction matches the actual label (if provided)
        comparison = "Correct" if actual_label and food_name.lower() == actual_label.lower() else "Incorrect"

        # Get top 3 predictions for additional insights (e.g., "Did you mean...?")
        # np.argsort returns indices that would sort the array. [-3:][::-1] gets top 3 descending.
        top_3_indices = np.argsort(raw_output[0])[-3:][::-1]
        top_3_predictions = []
        for idx in top_3_indices:
            if idx < len(class_names): # Ensure index is valid
                top_3_predictions.append({
                    "class": class_names[idx],
                    "confidence": f"{raw_output[0][idx]*100:.2f}%"
                })

        # Prepare the prediction response
        response = {
            "low_confidence": False,
            "food_name": food_name,
            "confidence": f"{confidence_percentage:.2f}%",
            "comparison": comparison, # Only relevant if actual_label is provided
            "top_3_predictions": top_3_predictions,
            "message": f"Food identified successfully with {confidence_percentage:.2f}% confidence."
        }

        # Fetch nutrition information from the database
        nutrition = get_nutrition_from_db(food_name)
        if "error" not in nutrition:
            # If nutrition data is found, add it to the response
            response.update(nutrition)
        else:
            # If nutrition data is not found or there's a DB error, add an error key
            response["nutrition_error"] = nutrition["error"]

        return response

    except Exception as e:
        print(f"❌ Critical error during prediction: {e}")
        # Print full traceback for debugging server-side issues
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error during prediction: {str(e)}"}), 500

# --- Nutrition Data Fetching Function ---
def get_nutrition_from_db(food_name):
    """
    Fetches nutrition details (carbs, protein, fat, calories)
    for the predicted food from the MySQL database.
    Includes case-insensitive search.
    """
    conn = get_db_connection()
    if conn is None:
        return {"error": "Failed to connect to nutrition database."}

    try:
        # Use dictionary=True to get results as dictionaries (column_name: value)
        cursor = conn.cursor(dictionary=True) 
        
        # Attempt exact match first
        query = "SELECT carbs, protein, fat, calories FROM food_entries WHERE food_name = %s"
        cursor.execute(query, (food_name,))
        result = cursor.fetchone()
        
        if result:
            print("✅ Nutrition Data Found:", result)
            return result
        else:
            print(f"❌ Nutrition Data Not Found for: '{food_name}' (exact match). Trying case-insensitive.")
            # If no exact match, try case-insensitive search
            query = "SELECT carbs, protein, fat, calories FROM food_entries WHERE LOWER(food_name) = LOWER(%s)"
            cursor.execute(query, (food_name,))
            result = cursor.fetchone()
            
            if result:
                print("✅ Nutrition Data Found (case-insensitive):", result)
                return result
            else:
                print(f"❌ Nutrition Data Not Found for: '{food_name}' (case-insensitive).")
                return {"error": f"Nutrition data for '{food_name}' not found in the database."}
                
    except mysql.connector.Error as db_err:
        print(f"❌ Database query error fetching nutrition: {db_err}")
        return {"error": f"Database query failed: {str(db_err)}"}
    except Exception as e:
        print(f"❌ Unexpected error fetching nutrition: {e}")
        return {"error": f"An unexpected error occurred while fetching nutrition: {str(e)}"}
    finally:
        # Ensure the database connection is closed
        if conn:
            conn.close()
            print("✅ Database connection closed.")

# --- API Route to Handle Food Image Uploads ---
@app.route("/food", methods=["POST", "OPTIONS"])
def predict_food():
    # Handle CORS pre-flight requests (OPTIONS method)
    if request.method == "OPTIONS":
        # The CORS headers are handled by Flask-CORS or the @app.after_request decorator.
        # This just needs to return a 200 OK for pre-flight.
        return "", 200
        
    # Check if a file was sent in the request
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Please include an image file."}), 400

    file = request.files["file"]
    
    # Check if the file field is empty (no file selected)
    if file.filename == "":
        return jsonify({"error": "No selected file. Please choose an image to upload."}), 400
        
    # Validate file type based on allowed extensions
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"Invalid file type. Allowed formats are: {', '.join(allowed_extensions)}"}), 400

    # Get optional 'actual_label' from form data (useful for testing/evaluation)
    actual_label = request.form.get("actual_label")
    print(f"📥 Received file: '{file.filename}', with optional actual_label: '{actual_label if actual_label else 'N/A'}'")

    # Call the prediction function
    prediction_result = predict_food_from_image(file, actual_label)
    
    print(f"📤 Sending response: {prediction_result}")
    # Return the prediction result as JSON
    return jsonify(prediction_result)

# --- Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health_check():
    """
    A simple endpoint to check if the server is running and the model is loaded.
    """
    db_status = "unreachable"
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            db_status = "reachable"
            conn.close()
    except Exception as e:
        print(f"Health check DB error: {e}")
        db_status = f"error ({str(e)})"
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "database_status": db_status,
        "classes_loaded": len(class_names),
        "class_names_source": "dynamic" if os.path.exists(DATASET_PATH) else "default"
    })

# --- Main execution block ---
if __name__ == "__main__":
    print("\n--- Flask Server Setup ---")
    print("✅ Registered Routes:")
    # Print all routes and their allowed methods for verification
    for rule in app.url_map.iter_rules():
        # Exclude static route which is often generated by Flask
        if 'static' not in rule.endpoint:
            print(f"  - {rule.endpoint}: {rule.rule} {list(rule.methods)}")
    
    print(f"\n🚀 Starting Flask server on http://127.0.0.1:5000/")
    print(f"Model loaded status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    print(f"Number of food classes recognized: {len(class_names)}")
    # Run the Flask app. debug=True allows for auto-reloading and better error messages during development.
    app.run(debug=True, port=5000)
