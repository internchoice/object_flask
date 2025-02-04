import json
import os
import cv2
import torch
import numpy as np
import pyttsx3
import pytesseract
import psutil
import queue
import threading
from flask import Flask, render_template, Response, jsonify, request
from scipy.spatial.distance import euclidean
from gtts import gTTS
import pygame
import time
import requests
import face_recognition
import os

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

# Load known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0]) 
# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speech_queue = queue.Queue()

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLOv5s model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cpu')

# Start background speech thread
def _run_speech():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            while not speech_queue.empty():  # Process all queued messages
                next_text = speech_queue.get()
                print(f"Speaking: {next_text}")  # Debugging output
                engine.say(next_text)
                engine.runAndWait()
        except RuntimeError as e:
            print(f"Speech error: {e}")

speech_thread = threading.Thread(target=_run_speech, daemon=True)
speech_thread.start()

spoken_messages = {}  # Dictionary to track when messages were last spoken
SPEECH_DELAY = 5  # Minimum seconds before repeating a message

import os
from gtts import gTTS
import pygame

pygame.mixer.init()

def speak(text):
    """Queue text for speech processing."""
    speech_queue.put(text)
threading.Thread(target=speak, args=("Object detected",)).start()
def speak_text(text):
    """Generate speech using gTTS and play it using pygame."""
    try:
        # Generate speech using gTTS
        tts = gTTS(text=text, lang="en")
        filename = "temp_audio.mp3"
        tts.save(filename)

        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
            time.sleep(0.1)

        pygame.mixer.quit()  # Ensure pygame quits properly
        os.remove(filename)  # Remove file after playing
    except Exception as e:
        print(f"Speech error: {e}")

# Environment detection
ENVIRONMENTS = {
    "street": ["car", "traffic light", "bus", "truck"],
    "park": ["tree", "bench", "dog", "bird"],
    "indoors": ["person", "chair", "table", "tv", "sofa"]
}
current_environment = None

# Constants
FRAME_SKIP = 5
TEXT_DETECTION_SKIP = FRAME_SKIP * 3
RESIZE_FACTOR = 0.5

def detect_environment(labels):
    """Determine the likely environment based on detected objects."""
    global current_environment
    environment_counts = {env: 0 for env in ENVIRONMENTS}
    
    for label in labels:
        for env, objects in ENVIRONMENTS.items():
            if label in objects:
                environment_counts[env] += 1

    detected_environment = max(environment_counts, key=environment_counts.get)
    
    if environment_counts[detected_environment] > 0 and detected_environment != current_environment:
        current_environment = detected_environment
        speak_text(f"Environment detected: {current_environment}")
        print(f"Environment detected: {current_environment}")

recently_detected_objects = {}
DETECTION_TIMEOUT = 5  # Avoid repeating the same object within X seconds

def detect_objects(frame):
    """Detect objects and provide audio feedback only when new objects appear."""
    small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        results = model([small_frame])
    except Exception as e:
        print(f"Error with YOLOv5 model: {e}")
        return [], frame

    detections = results.pandas().xyxy[0]
    if detections.empty:
        return [], frame

    labels = []
    current_time = time.time()

    for _, row in detections.iterrows():
        class_name = row['name']
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        labels.append(class_name)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Speak only if the object hasn't been detected recently
        if class_name not in recently_detected_objects or current_time - recently_detected_objects[class_name] > DETECTION_TIMEOUT:
            recently_detected_objects[class_name] = current_time
            speak_text(f"Detected {class_name}")

    detect_environment(labels)
    return labels, frame

def preprocess_for_text_detection(frame):
    """Preprocess image for better text detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Additional steps like dilation to highlight text areas
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    return dilated

def detect_text(frame):
    """Detect text in the frame and provide audio feedback."""
    processed_frame = preprocess_for_text_detection(frame)
    
    # Detect text using Tesseract
    h, w = processed_frame.shape
    boxes = pytesseract.image_to_boxes(processed_frame)

    if boxes.strip():  # Only proceed if there are boxes detected
        for b in boxes.splitlines():
            b = b.split()
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(frame, (x, h), (w, y), (0, 255, 0), 2)  # Draw box around detected text

        text = pytesseract.image_to_string(processed_frame, config='--psm 6')
        
        if text.strip():
            print("Detected text:", text)
            speak_text(f"Text detected: {text}")
        else:
            print("No text detected in this frame.")
            
def check_battery():
    """Check battery level and provide alert if low."""
    battery = psutil.sensors_battery()
    if battery and battery.percent % 10 == 0:
        speak_text(f"Battery level at {battery.percent} percent.")
        if battery.percent < 20 and not battery.power_plugged:
            speak("Battery low, please charge your device.")

recently_detected_faces = {}

def recognize_faces(frame):
    """Detect and recognize faces while avoiding repeated announcements."""
    global recently_detected_faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_time = time.time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_face_names[matched_idx]

        # Draw bounding box and name
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Avoid repeating names too frequently
        if name not in recently_detected_faces or current_time - recently_detected_faces[name] > 10:
            recently_detected_faces[name] = current_time
            speak_text(f"this is {name}")

    return frame

def gen_frames():
    """Capture and process video frames."""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        labels, frame = detect_objects(frame)
        recognize_faces(frame)
        
        # Process text detection on every frame
        detect_text(frame)

        # You can add the battery check or environment detection here if needed
        check_battery()

        # Encode frame to send to browser
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/location', methods=['GET'])
def get_location():
    """Fetch user's approximate location using an external API."""
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        if 'loc' in data:
            lat, lon = map(float, data['loc'].split(','))
            return jsonify({'latitude': lat, 'longitude': lon})
    except Exception as e:
        print(f"Error fetching location: {e}")
    return jsonify({'error': 'Unable to get location'}), 500

@app.route('/track_location')
def track_location():
    return render_template('loc.html')

@app.route('/update_location', methods=['POST'])
def update_location():
    data = request.get_json()
    current_location['latitude'] = data['latitude']
    current_location['longitude'] = data['longitude']
    return jsonify({"status": "Location updated"}), 200

def location_generator():
    """Continuously fetch and stream location data."""
    while True:
        try:
            response = requests.get('https://ipinfo.io/json')
            data = response.json()
            if 'loc' in data:
                lat, lon = map(float, data['loc'].split(','))
                latest_location = f"Latitude: {lat}, Longitude: {lon}"
                yield f"data: {latest_location}\n\n"
            else:
                yield "data: Unable to get location\n\n"
        except Exception as e:
            yield "data: Error fetching location\n\n"
        time.sleep(5)

current_location = {"latitude": 18.5196, "longitude": 73.8554}

@app.route('/location_stream')
def location_stream():
    def generate_location():
        while True:
            yield f"data: {json.dumps(current_location)}\n\n"
            time.sleep(1)

    return Response(generate_location(), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
