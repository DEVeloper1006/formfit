import os
import cv2
import numpy as np
import mediapipe as mp

input_dir = "data"
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, smooth_landmarks=True)

# Define joint triplets for angle calculations
joints_triplets = [
    (11, 13, 15),  # Left shoulder, elbow, wrist
    (12, 14, 16),  # Right shoulder, elbow, wrist
    (23, 25, 27),  # Left hip, knee, ankle
    (24, 26, 28),  # Right hip, knee, ankle
    (11, 23, 25),  # Left shoulder, hip, knee
    (12, 24, 26),  # Right shoulder, hip, knee
]

def calculate_angle (p1, p2, p3):
    vector1 = p1 - p2
    vector2 = p3 - p2
    dot_product = np.dot(vector1, vector2)
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    epsilon = 1e-6
    cosine_angle = dot_product / (magnitude + epsilon)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def process_video(file_path, output_path):
    cap = cv2.VideoCapture(file_path)
    joint_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 256x256
        frame_resized = cv2.resize(frame, (256, 256))

        # Process with MediaPipe
        results = pose.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            
            for j1, j2, j3 in joints_triplets:
                angle = calculate_angle(landmarks[j1], landmarks[j2], landmarks[j3])
                joint_angles.append(angle)
            
    # Convert to NumPy array
    joint_array = np.array(joint_angles)

    # Save to .npy file
    np.save(output_path, joint_array)
    cap.release()

for file_name in os.listdir(input_dir):
    if file_name.endswith(".mp4"):
        file_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace(".mp4", ".npy"))
        if os.path.isfile(output_path):
            print(f"Skipping {file_name} as output file already exists.")
        else:
            process_video(file_path, output_path)
            print(f"Processed {file_name}")