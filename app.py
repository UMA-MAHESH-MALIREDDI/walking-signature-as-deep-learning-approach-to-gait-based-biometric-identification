from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Force eager execution
import os
import cv2
import numpy as np
import mediapipe as mp


app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB file limit

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Paths
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = "gait_model.h5"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load or create model
if os.path.exists(MODEL_PATH):
    print("Loading pre-trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    # Recompile the model with a new optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Reinitialize optimizer
        loss='binary_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )
else:
    print("Creating new model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(33, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )

# Enable debug mode for tf.data
tf.data.experimental.enable_debug_mode()

def normalize_features(features, target_length=100):
    """
    Normalize features to a fixed length by truncating or padding.
    """
    if features.shape[0] > target_length:
        # Truncate excess frames
        return features[:target_length]
    elif features.shape[0] < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - features.shape[0], 33, 2))
        return np.concatenate([features, padding], axis=0)
    else:
        return features

def extract_gait_features(video_path):
    """Extract pose keypoints from video using MediaPipe"""
    cap = cv2.VideoCapture(video_path)
    features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            features.append(keypoints)
    
    cap.release()
    features = np.array(features, dtype=np.float32) if features else None
    
    # Normalize features to a fixed length
    if features is not None:
        features = normalize_features(features, target_length=100)
    
    return features

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        print("Upload request received")
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        name = request.form.get('name', '').strip()
        
        print(f"Received file: {file.filename}, name: {name}")
        
        # Validate inputs
        if not name:
            print("Name is missing")
            return jsonify({"error": "Name is required"}), 400
        if file.filename == '':
            print("No file selected")
            return jsonify({"error": "No file selected"}), 400
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            print("Invalid file type")
            return jsonify({"error": "Only MP4/MOV/AVI files allowed"}), 400

        # Save file
        filename = f"{name}_{file.filename}"
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '.')).rstrip()
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        
        print(f"Saving file to: {video_path}")
        file.save(video_path)
        
        # Extract features
        print("Extracting features...")
        features = extract_gait_features(video_path)
        if features is None or len(features) == 0:
            print("No poses detected")
            os.remove(video_path)  # Clean up invalid file
            return jsonify({"error": "No human poses detected in video"}), 400
            
        # Save features
        feature_path = os.path.join(UPLOAD_FOLDER, f"{name}_features.npy")
        np.save(feature_path, features.astype(np.float32))
        print(f"Saved features for {name} with shape {features.shape}")
        
        # Return success response
        response_data = {"message": f"Successfully uploaded {name}'s data!"}
        print("Sending response:", response_data)  # Debug log
        return jsonify(response_data)

    except Exception as e:
        print(f"Upload failed: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        print("\n=== Starting Model Training ===")
        
        # Load all valid features
        features_list = []
        valid_users = 0
        
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith("_features.npy"):
                try:
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    features = np.load(filepath)
                    
                    # Validate features
                    if features.ndim != 3 or features.shape[1:] != (33, 2):
                        print(f"Skipping invalid features: {filename}")
                        continue
                    if np.isnan(features).any():
                        print(f"Skipping features with NaN values: {filename}")
                        continue
                        
                    features_list.append(features)
                    valid_users += 1

                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    continue

        if valid_users == 0:
            return jsonify({"error": "No valid training data found"}), 400
            
        # Prepare dataset
        X = np.concatenate(features_list, axis=0)
        y = np.ones((X.shape[0], 1))  # Dummy labels
        
        print(f"\n=== Training Summary ===")
        print(f"Total samples: {X.shape[0]}")
        print(f"Input shape: {X.shape[1:]}")
        
        # Reinitialize model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(33, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True
        )
        
        # Train model
        print("\nTraining model...")
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        model.save(MODEL_PATH)
        
        return jsonify({
            "message": f"Model trained successfully with {valid_users} users!",
            "samples": X.shape[0],
            "input_shape": f"{X.shape[1:]}"
        })

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500
    
@app.route('/predict_gait', methods=['POST'])
def predict_gait():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save and process the video
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        features = extract_gait_features(file_path)
        
        if features is None:
            return jsonify({"error": "No poses detected"}), 400

        # Normalize features
        features = normalize_features(features, target_length=100)

        # Find the closest matching user from training data
        best_match = None
        max_similarity = -1

        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith("_features.npy"):
                user_name = filename.split("_")[0]
                user_features = np.load(os.path.join(UPLOAD_FOLDER, filename))
                
                # Calculate similarity (simple average difference)
                similarity = -np.mean(np.abs(features - user_features))
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = user_name

        if best_match:
            return jsonify({"user": best_match})
        else:
            return jsonify({"error": "No matching user found"}), 400

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)