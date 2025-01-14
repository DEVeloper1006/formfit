import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scalers
model = load_model("model/best_model.keras")
scaler = joblib.load("model/scaler_train.joblib")

# Define functions to extract features
def calculate_angle(vec1, vec2):
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def process_video(video_path):
    """Extract features from the new video."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, smooth_landmarks=True)
    cap = cv2.VideoCapture(video_path)

    angles_data = []
    coordinates_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            rs = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])  # Right Shoulder
            ls = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])  # Left Shoulder
            le = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])  # Left Elbow
            lw = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])  # Left Wrist
            li = np.array([landmarks[19].x, landmarks[19].y, landmarks[19].z])  # Left Index Finger

            # Compute angles
            ls_le = le - ls
            le_lw = lw - le
            lw_li = li - lw

            shoulder_angle = calculate_angle(rs - ls, ls_le)
            elbow_angle = calculate_angle(ls_le, le_lw)
            wrist_angle = calculate_angle(le_lw, lw_li)

            # Append angles and coordinates
            angles_data.append([shoulder_angle, elbow_angle, wrist_angle])
            coordinates_data.append(np.concatenate([rs, ls, le, lw, li]))

    cap.release()
    pose.close()

    # Combine data into a DataFrame
    angles_df = pd.DataFrame(angles_data, columns=["Left Shoulder", "Left Elbow", "Left Wrist"])
    coordinates_df = pd.DataFrame(
        coordinates_data,
        columns=[
            "RS_x", "RS_y", "RS_z",
            "LS_x", "LS_y", "LS_z",
            "LE_x", "LE_y", "LE_z",
            "LW_x", "LW_y", "LW_z",
            "LI_x", "LI_y", "LI_z",
        ]
    )
    final_df = pd.concat([angles_df, coordinates_df], axis=1)
    return final_df

# Sliding window generation
def create_sliding_windows(data, window_size=2, step_size=1):
    """
    Generate sliding windows from the data.
    Each window has 50 frames worth of features.
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size].to_numpy().flatten()
        windows.append(window)
    return np.array(windows)

# Predict new video
def predict_video(video_path):
    # Step 1: Extract features
    features_df = process_video(video_path)
    
    # Step 2: Convert features to a NumPy array (drop column names)
    features_array = features_df.to_numpy()

    # Step 3: Create sliding windows
    sliding_windows = create_sliding_windows(
        pd.DataFrame(features_array), 
        window_size=50,  # Match the training window size
        step_size=25     # Match the training step size
    )

    # Step 4: Standardize features using the saved scaler
    sliding_windows_flat = sliding_windows.reshape(sliding_windows.shape[0], -1)  # Flatten windows
    features_scaled = scaler.transform(sliding_windows_flat)  # Standardize

    # Step 5: Reshape for LSTM input (samples, timesteps, features)
    input_data = features_scaled.reshape((-1, 50, features_df.shape[1]))  # (samples, timesteps, features)

    # Step 6: Make predictions
    predictions = model.predict(input_data)
    predicted_classes = np.argmax(predictions, axis=1)

    # Step 7: Aggregate predictions
    final_prediction = np.bincount(predicted_classes).argmax()  # Majority vote
    return final_prediction


# Example usage
video_path = "data/3_12_cam2.mp4"  # Replace with the user's video file
predicted_class = predict_video(video_path)
print(f"Predicted Exercise Class: {predicted_class}")
