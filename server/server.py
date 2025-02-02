import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
import joblib
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Load the pre-trained model and scaler
model = tf.keras.models.load_model("model/best_model.keras")
scaler = joblib.load("model/scaler_train.joblib")

def quaternion_angle(vec1, vec2):
    """Calculate the angle between two vectors using quaternion representation."""
    cross_prod = np.cross(vec1, vec2)
    dot_prod = np.dot(vec1, vec2)
    angle = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
    return np.degrees(angle)

def extract_features(landmarks):
    """Extract features (angles and coordinates) from pose landmarks."""
    rs = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])  # Right Shoulder
    ls = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])  # Left Shoulder
    le = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])  # Left Elbow
    lw = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])  # Left Wrist
    li = np.array([landmarks[19].x, landmarks[19].y, landmarks[19].z])  # Left Index Finger

    # Compute vectors
    ls_le = le - ls  # Left Shoulder to Left Elbow
    le_lw = lw - le  # Left Elbow to Left Wrist
    lw_li = li - lw  # Left Wrist to Left Index Finger

    # Calculate angles (quaternion-based)
    shoulder_angle = quaternion_angle(rs - ls, ls_le)
    elbow_angle = quaternion_angle(ls_le, le_lw)
    wrist_angle = quaternion_angle(le_lw, lw_li)

    # Combine angles and coordinates into a feature vector
    features = np.concatenate([
        [shoulder_angle, elbow_angle, wrist_angle],  # Angles
        rs, ls, le, lw, li  # Coordinates
    ])
    return features

class FrameBuffer:
    def __init__(self, model, scaler):
        self.buffer = []
        self.model = model
        self.scaler = scaler
        self.frame_count = 0
    
    def add_frame(self, frame):
        """Process frame for both landmarks and exercise classification."""
        # Process landmarks for current frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        response_data = {}
        
        if results.pose_landmarks:
            # Draw landmarks on frame copy
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Encode annotated frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            
            response_data = {
                "landmarks_detected": True,
                "annotated_frame": base64_frame
            }
        
        # Add frame to buffer for exercise classification
        self.buffer.append(frame)
        self.frame_count += 1
        
        # Process exercise classification if buffer is full
        if len(self.buffer) == 50:
            exercise_class = self.process_buffer()
            if exercise_class is not None:
                response_data["exercise_class"] = int(exercise_class)
            
            self.buffer = []
            self.frame_count = 0
        
        return response_data
    
    def process_buffer(self):
        """Process the 50-frame buffer for exercise classification."""
        if len(self.buffer) != 50:
            return None

        feature_vectors = []
        for frame in self.buffer:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                features = extract_features(results.pose_landmarks.landmark)
                feature_vectors.append(features)
            else:
                print("⚠️ No landmarks detected in a frame. Skipping buffer.")
                return None

        feature_vectors = np.array(feature_vectors)
        feature_vectors_reshaped = feature_vectors.reshape(1, -1)
        feature_vectors_scaled = self.scaler.transform(feature_vectors_reshaped)
        feature_vectors_scaled = feature_vectors_scaled.reshape(1, 50, 18)

        prediction = self.model.predict(feature_vectors_scaled)
        exercise_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted Exercise Class: {exercise_class}")
        return exercise_class

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_buffer = FrameBuffer(model, scaler)
    
    try:
        while True:
            data = await websocket.receive_text()
            packet = json.loads(data)

            if not all(key in packet for key in ["frame", "width", "height"]):
                print("Invalid frame format")
                continue

            try:
                frame_bytes = base64.b64decode(packet["frame"])
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Invalid image data")

                if img.shape[1] != packet["width"] or img.shape[0] != packet["height"]:
                    img = cv2.resize(img, (packet["width"], packet["height"]))

                # Process frame for both landmarks and exercise classification
                response_data = frame_buffer.add_frame(img)
                await websocket.send_text(json.dumps(response_data))

            except Exception as e:
                print(f"🚨 Frame processing error: {str(e)}")
                continue
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            print("WebSocket already closed")