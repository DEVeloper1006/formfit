from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import os
import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Disable GPU and set TF logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Constants
BUFFER_SIZE = 30
EXPECTED_DIMENSIONS = (256, 256)

@dataclass
class ClientState:
    frame_buffer: List[np.ndarray]  # Buffer for raw frames
    coordinate_buffer: List[Tuple[int, List[Tuple[float, float, float]]]]  # Buffer for (frame_num, coordinates)
    current_exercise: Optional[str]
    is_initially_classified: bool
    frame_count: int

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """Process a single frame and return pose landmarks."""
        try:
            if frame is None or not isinstance(frame, np.ndarray):
                logging.error(f"Invalid frame type: {type(frame)}")
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                return landmarks
            return None

        except Exception as e:
            logging.error(f"Error in pose estimation: {e}")
            return None

    def __del__(self):
        self.pose.close()

class ExerciseProcessor:
    @staticmethod
    def classify_exercise(coordinate_sequence: List[Tuple[int, List[Tuple[float, float, float]]]]) -> str:
        """
        Placeholder for LSTM-based exercise classification using sequence of coordinates.
        Replace with your LSTM classification logic.
        """
        # Sort by frame number to ensure temporal order
        sorted_sequence = sorted(coordinate_sequence, key=lambda x: x[0])
        coordinates_only = [coords for _, coords in sorted_sequence]
        
        logging.info(f"Classifying sequence of {len(coordinates_only)} coordinate sets")
        return "exercise_type"  # Replace with actual LSTM classification

    @staticmethod
    def apply_rl_policy(coordinates: List[Tuple[float, float, float]], 
                       exercise_type: str) -> Dict:
        """
        Placeholder for RL policy application.
        Replace with your RL logic.
        """
        return {
            "feedback": "Placeholder feedback",
            "score": 0.0,
            "corrections": []
        }

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.clients: Dict[str, ClientState] = {}
        self.pose_estimator = PoseEstimator()
        self.exercise_processor = ExerciseProcessor()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    def initialize_client(self, client_id: str):
        """Initialize client state."""
        self.clients[client_id] = ClientState(
            frame_buffer=[],
            coordinate_buffer=[],
            current_exercise=None,
            is_initially_classified=False,
            frame_count=0
        )

    def cleanup_client(self, client_id: str):
        """Clean up client data."""
        self.clients.pop(client_id, None)

app = FastAPI()
manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time frame processing."""
    await manager.connect(websocket)
    manager.initialize_client(client_id)
    client_state = manager.clients[client_id]
    
    try:
        while True:
            # Receive and decode frame
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Failed to decode frame"})
                continue

            # Add frame to buffer and increment counter
            client_state.frame_buffer.append(frame)
            client_state.frame_count += 1

            # Get coordinates for current frame
            coordinates = manager.pose_estimator.process_frame(frame)
            if coordinates:
                # Store coordinates with frame number
                client_state.coordinate_buffer.append((client_state.frame_count, coordinates))

            # Prepare base response
            response = {
                "event": "frame_received",
                "frame_count": client_state.frame_count,
                "buffer_size": len(client_state.frame_buffer),
            }

            if not client_state.is_initially_classified:
                if len(client_state.coordinate_buffer) >= BUFFER_SIZE:
                    # Initial classification using sequence of coordinates
                    logging.info(f"Client {client_id}: Performing initial classification with {BUFFER_SIZE} coordinate sets")
                    client_state.current_exercise = manager.exercise_processor.classify_exercise(
                        client_state.coordinate_buffer
                    )
                    client_state.is_initially_classified = True
                    
                    # Clear buffers
                    client_state.frame_buffer.clear()
                    client_state.coordinate_buffer.clear()
                    
                    response.update({
                        "event": "initial_classification",
                        "exercise": client_state.current_exercise,
                        "coordinates": coordinates if coordinates else None
                    })
                else:
                    response.update({
                        "event": "buffering_for_classification",
                        "frames_needed": BUFFER_SIZE - len(client_state.coordinate_buffer)
                    })
            else:
                # After initial classification, process each frame through RL
                if coordinates:
                    logging.info(f"Client {client_id}: Applying RL policy for exercise: {client_state.current_exercise}")
                    rl_feedback = manager.exercise_processor.apply_rl_policy(
                        coordinates, 
                        client_state.current_exercise
                    )
                    
                    response.update({
                        "event": "frame_processed",
                        "exercise": client_state.current_exercise,
                        "rl_feedback": rl_feedback
                    })

                # Check for reclassification when coordinate buffer is full
                if len(client_state.coordinate_buffer) >= BUFFER_SIZE:
                    logging.info(f"Client {client_id}: Performing reclassification with {BUFFER_SIZE} coordinate sets")
                    new_exercise = manager.exercise_processor.classify_exercise(
                        client_state.coordinate_buffer
                    )
                    
                    if new_exercise != client_state.current_exercise:
                        client_state.current_exercise = new_exercise
                        response.update({
                            "event": "exercise_changed",
                            "new_exercise": new_exercise,
                            "coordinates": coordinates if coordinates else None
                        })
                    
                    # Clear buffers after classification
                    client_state.frame_buffer.clear()
                    client_state.coordinate_buffer.clear()

            # Send response for every frame
            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        manager.cleanup_client(client_id)
        logging.info(f"Client {client_id} disconnected")
    except Exception as e:
        logging.error(f"Error with client {client_id}: {e}")
        manager.cleanup_client(client_id)
        await websocket.close()
