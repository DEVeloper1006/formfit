"use client";

import { useState, useRef, useEffect } from "react";

const VideoRecorder = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [feedback, setFeedback] = useState("");
  const intervalRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);

  const reconnectWebSocket = () => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws");

    wsRef.current.onopen = () => {
      console.log("WebSocket connected");
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.feedback) {
        setFeedback(data.feedback);
      }

      if (data.annotated_frame) {
        const img = new Image();
        img.src = "data:image/jpeg;base64," + data.annotated_frame;
        img.onload = () => {
          const overlayCanvas = overlayRef.current;
          if (!overlayCanvas) return;
          const ctx = overlayCanvas.getContext("2d");
          overlayCanvas.width = videoRef.current.videoWidth;
          overlayCanvas.height = videoRef.current.videoHeight;
          ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
          ctx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
        };
      }
    };

    wsRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
  };

  useEffect(() => {
    reconnectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const startCamera = async () => {
    if (isCameraOn) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setIsCameraOn(true);
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      videoRef.current.srcObject = null;
      setIsCameraOn(false);
      stopRecording();
    }
  };

  const captureAndSendFrame = () => {
    if (feedback === "Reposition Camera") {
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const fullBase64 = canvas.toDataURL("image/jpeg", 0.8);
    const base64Data = fullBase64.split(",")[1];

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          frame: base64Data,
          width: canvas.width,
          height: canvas.height,
        })
      );
    }
  };

  const startRecording = () => {
    if (!isCameraOn || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      reconnectWebSocket();
    }
    setIsRecording(true);
    intervalRef.current = setInterval(captureAndSendFrame, 100);
  };

  const stopRecording = () => {
    setFeedback("");
    setIsRecording(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    const overlayCanvas = overlayRef.current;
    if (overlayCanvas) {
      const ctx = overlayCanvas.getContext("2d");
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">FormFit Web Prototype</h1>
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-2xl">
        <div className="relative">
          <video ref={videoRef} autoPlay muted className="w-full h-auto rounded-lg mb-4" />
          <canvas ref={canvasRef} style={{ display: "none" }} />
          <canvas ref={overlayRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
        </div>
        <div className="flex justify-center space-x-4 mb-4">
          <button
            onClick={() => (isCameraOn ? stopCamera() : startCamera())}
            className={`px-6 py-2 rounded-lg text-white font-semibold ${
              isCameraOn ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
            }`}
          >
            {isCameraOn ? "Turn Off Camera" : "Turn On Camera"}
          </button>
        </div>
        <div className="flex justify-center space-x-4">
          <button
            onClick={startRecording}
            disabled={isRecording || !isCameraOn}
            className={`px-6 py-2 rounded-lg text-white font-semibold ${
              isRecording || !isCameraOn ? "bg-gray-400 cursor-not-allowed" : "bg-blue-500 hover:bg-blue-600"
            }`}
          >
            Start Capturing
          </button>
          <button
            onClick={stopRecording}
            disabled={!isRecording}
            className={`px-6 py-2 rounded-lg text-white font-semibold ${
              !isRecording ? "bg-gray-400 cursor-not-allowed" : "bg-red-500 hover:bg-red-600"
            }`}
          >
            Stop Capturing
          </button>
        </div>
        {feedback && (
          <div className="mt-4 text-center">
            <p
              className={`text-lg font-semibold ${
                feedback === "Position Correct" ? "text-green-600" : "text-red-600"
              }`}
            >
              {feedback}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoRecorder;
