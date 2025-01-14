import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os

def calculate_angle(vec1, vec2):
    """Calculate the angle between two vectors using the dot product."""
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to handle precision issues
    return np.degrees(angle)


def quaternion_angle(vec1, vec2):
    """Calculate the angle between two vectors using quaternion representation."""
    cross_prod = np.cross(vec1, vec2)
    dot_prod = np.dot(vec1, vec2)
    angle = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
    return np.degrees(angle)

def process_video(video_path, output_csv):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, smooth_landmarks=True, model_complexity=1, enable_segmentation=True)
    cap = cv2.VideoCapture(video_path)

    angles_data = []
    coordinates_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get the required landmarks
            rs = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])  # Right Shoulder
            ls = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])  # Left Shoulder
            le = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])  # Left Elbow
            lw = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])  # Left Wrist
            li = np.array([landmarks[19].x, landmarks[19].y, landmarks[19].z])  # Left Index Finger

            # Compute vectors
            ls_le = le - ls  # Left Shoulder to Left Elbow
            le_lw = lw - le  # Left Elbow to Left Wrist
            lw_li = li - lw  # Left Wrist to Left Index Finger

            # Calculate angles (plane-based)
            shoulder_angle = calculate_angle(rs - ls, ls_le)
            elbow_angle = calculate_angle(ls_le, le_lw)
            wrist_angle = calculate_angle(le_lw, lw_li)

            # Alternative: Calculate angles (quaternion-based)
            shoulder_angle_q = quaternion_angle(rs - ls, ls_le)
            elbow_angle_q = quaternion_angle(ls_le, le_lw)
            wrist_angle_q = quaternion_angle(le_lw, lw_li)

            # Append angles to the data
            # angles_data.append([shoulder_angle, elbow_angle, wrist_angle])
            angles_data.append([shoulder_angle_q, elbow_angle_q, wrist_angle_q])
            # Append raw XYZ coordinates
            coordinates_data.append(np.concatenate([rs, ls, le, lw, li]))

    # Write to CSV files
    angles_df = pd.DataFrame(angles_data, columns=["Left Shoulder", "Left Elbow", "Left Wrist"])

    coordinates_df = pd.DataFrame(
        coordinates_data,
        columns=[
            "RS_x", "RS_y", "RS_z",
            "LS_x", "LS_y", "LS_z",
            "LE_x", "LE_y", "LE_z",
            "LW_x", "LW_y", "LW_z",
            "LI_x", "LI_y", "LI_z",
        ],
    )
    
    final_df = pd.concat([angles_df, coordinates_df], axis=1)
    final_df.to_csv(output_csv, index=False)

    cap.release()
    pose.close()
    print(f"Angles and Coordinates saved to {output_csv}")

input_path = "data"
save_path = "final_data"
os.makedirs(save_path, exist_ok=True)

for file_name in os.listdir(input_path):
    if file_name.endswith(".mp4"):
        file_prefix = file_name.replace(".mp4", "")
        parts = file_prefix.split("_")
        exercises = ["09", "10", "11", "12"]
        if (parts[1] in exercises):
            file_path = os.path.join(input_path, file_name)
            output_path = os.path.join(save_path, file_name.replace(".mp4", ".npy"))
            if os.path.isfile(output_path):
                print(f"Skipping {file_name} as output file already exists.")
            else:
                process_video(video_path=file_path, output_csv=output_path)
                print(f"Processed {file_name}")
