from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2 as cv
import numpy as np
import math

app = FastAPI()

protoFile = "body_25/pose_deploy.prototxt"
weightsFile = "body_25/pose_iter_584000.caffemodel"
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24,
    "Background": 25
}

# Define pairs for angles calculation
ANGLE_PAIRS = [
    ("RShoulder", "RElbow", "RWrist"),
    ("LShoulder", "LElbow", "LWrist"),
    ("RHip", "RKnee", "RAnkle"),
    ("LHip", "LKnee", "LAnkle"),
    # Add more as needed
]

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points. Points are in (x, y) format."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    return angle

def process_frame(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    out = net.forward()
    
    points = {}
    angles = {}
    
    for part, i in BODY_PARTS.items():
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        if conf > 0.1:
            x = int((frameWidth * point[0]) / out.shape[3])
            y = int((frameHeight * point[1]) / out.shape[2])
            points[part] = (x, y)
        else:
            points[part] = None
    
    for triplet in ANGLE_PAIRS:
        p1, p2, p3 = triplet
        if points[p1] and points[p2] and points[p3]:
            angle = calculate_angle(points[p1], points[p2], points[p3])
            angles[f"{p1}_{p2}_{p3}"] = angle

    print(points)
    print(angles)
    
    return points, angles

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({'error': 'Image cannot be decoded'})
                continue

            keypoints, joint_angles = process_frame(frame)
            await websocket.send_json({'keypoints': keypoints, 'angles': joint_angles})
    except WebSocketDisconnect:
        await websocket.close()
