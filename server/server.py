import base64
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FrameBuffer:
    def __init__(self):
        self.buffer = []
    
    def add_frame(self, frame):
        self.buffer.append(frame)
        if len(self.buffer) >= 100:
            self.process_buffer()
    
    def process_buffer(self):
        print(f"✅ Processed batch of {len(self.buffer)} frames")
        self.buffer.clear()

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_buffer = FrameBuffer()
    
    try:
        while True:
            data = await websocket.receive_text()
            packet = json.loads(data)
            
            if not all(key in packet for key in ["frame", "width", "height"]):
                print("⚠️ Invalid frame format")
                continue

            try:
                # Decode base64
                frame_bytes = base64.b64decode(packet["frame"])
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                
                # Decode image
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Invalid image data")
                
                # Validate dimensions
                if img.shape[1] != packet["width"] or img.shape[0] != packet["height"]:
                    img = cv2.resize(img, (packet["width"], packet["height"]))
                
                # Add to buffer
                frame_buffer.add_frame(img)
                
            except Exception as e:
                print(f"🚨 Frame processing error: {str(e)}")
                continue

    except Exception as e:
        print(f"🔌 Connection error: {str(e)}")
    finally:
        await websocket.close()
        print("❌ Connection closed")