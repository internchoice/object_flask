from flask import Flask, render_template, request, Response, jsonify
import os
import cv2
import base64
import numpy as np
import threading
import time
import torch
import pyttsx3
import queue
import pytesseract  # Import pytesseract
import face_recognition

app = Flask(__name__)

cert_path = os.path.abspath("cert.pem")
key_path = os.path.abspath("key.pem")

global_frame = None
frame_lock = threading.Lock()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Create a queue for speech requests
speech_queue = queue.Queue()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prepare known faces for recognition
known_faces = {}
known_face_names = []
known_face_encodings = []

# Environment classification dictionary
ENVIRONMENTS = {
    "street": ["car", "traffic light", "bus", "truck"],
    "park": ["tree", "bench", "dog", "bird"],
    "indoors": ["person", "chair", "table", "tv", "sofa"]
}

current_environment = "unknown"

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_known_faces():
    """ Load known faces from the known_faces directory and prepare for recognition """
    global known_faces, known_face_names, known_face_encodings
    known_faces_dir = 'known_faces'  
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(known_faces_dir, filename)
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for face_recognition

            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

            for encoding in face_encodings:
                known_face_encodings.append(encoding)
                known_face_names.append(filename.split(".")[0])  # Using filename as the name

# Call this function to load the known faces on server startup
load_known_faces()

# Track if environment has been spoken
environment_spoken = False

def detect_environment(labels):
    """Determine the likely environment based on detected objects."""
    global current_environment, environment_spoken
    environment_counts = {env: 0 for env in ENVIRONMENTS}

    print(f"Detected labels: {labels}")  # Debug log to check detected labels

    for label in labels:
        for env, objects in ENVIRONMENTS.items():
            if label in objects:
                environment_counts[env] += 1

    detected_environment = max(environment_counts, key=environment_counts.get)

    if environment_counts[detected_environment] > 0 and detected_environment != current_environment:
        current_environment = detected_environment
        if not environment_spoken:  # Only speak if the environment has not been spoken yet
            speak_feedback(f"Environment detected: {current_environment}")
            environment_spoken = True
            print(f"Environment detected: {current_environment}")
        else:
            print(f"Environment remains: {current_environment}")
    elif detected_environment != current_environment:
        environment_spoken = False  # Reset if environment changes again

def detect_objects(frame):
    """ Detects objects using YOLOv5 and face recognition """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    
    labels = results.names
    detections = []
    face_names = []

    # Detect objects using YOLOv5
    for *box, conf, cls in results.xywh[0]:
        x1, y1, x2, y2 = map(int, box)
        class_name = labels[int(cls)]
        
        # Draw bounding box for objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        detections.append(class_name)

    # Detect environment based on detected labels
    detect_environment(detections)

    # Face detection and recognition
    faces = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, faces)
    
    for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
        recognized = False

        # Compare the face with known encodings
        for idx, known_encoding in enumerate(known_face_encodings):
            match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
            if match[0]:
                face_name = known_face_names[idx]
                cv2.putText(frame, face_name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                face_names.append(face_name)
                recognized = True
                break

        if not recognized:
            cv2.putText(frame, "Unknown", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Perform OCR to detect any text in the frame
    ocr_result = pytesseract.image_to_string(frame)
    process_text_signs(ocr_result)

    return frame, detections, face_names

def process_text_signs(ocr_text):
    """Process the OCR detected text and announce signs like 'EXIT' and 'STOP'."""
    ocr_text = ocr_text.upper()  # Convert text to uppercase for easy comparison
    if 'EXIT' in ocr_text:
        speak_feedback("Exit sign detected")
    elif 'STOP' in ocr_text:
        speak_feedback("Stop sign detected")

def speak_feedback(text):
    """ Add speech request to the queue. """
    speech_queue.put(text)

def process_speech():
    """ Process the speech queue in a background thread to ensure sequential processing. """
    while True:
        text = speech_queue.get()
        if text == "STOP":
            break
        engine.say(text)
        engine.runAndWait()

# Start a background thread to process the speech queue
speech_thread = threading.Thread(target=process_speech, daemon=True)
speech_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_frame', methods=["POST"])
def update_frame():
    """ Receives frames from the client, processes them and stores them """
    global global_frame
    data = request.json.get("frame")
    detections = []
    face_names = []
    environment = "unknown"  # Default environment

    if data:
        header, encoded = data.split(',', 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Store the received frame
        with frame_lock:
            global_frame = img

        # Detect objects and faces in the frame
        _, detections, face_names = detect_objects(global_frame)

        # Detect the environment based on objects
        environment = current_environment  # Assuming you already have current_environment variable in your backend

    return jsonify({'detections': detections, 'faces': face_names, 'environment': environment}), 200

@app.route('/processed_feed')
def processed_feed():
    """ Streams the processed client camera feed to the client """
    def generate():
        while True:
            with frame_lock:
                if global_frame is None:
                    continue
                
                frame_with_objects, _, _ = detect_objects(global_frame)

                # Encode the frame with object and face detection as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame_with_objects)
                if not ret:
                    continue

                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # Reduce lag by optimizing streaming timing
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=(cert_path, key_path))
