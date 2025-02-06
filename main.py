from flask import Flask, render_template, request, Response
import os
import cv2, base64, numpy as np
import threading
import time

app = Flask(__name__)

cert_path = os.path.abspath("cert.pem")
key_path = os.path.abspath("key.pem")

global_frame = None  # Stores the latest client frame (processed)
original_frame = None  # Stores the original client frame before processing
frame_lock = threading.Lock()

def display_client_stream():
    """ Continuously displays the received client stream (Processed) on the server. """
    global global_frame
    while True:
        if global_frame is not None:
            with frame_lock:
                cv2.imshow("Processed (Black & White) Client Stream on Server", global_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

# Start a separate thread for displaying the processed client stream on the server
threading.Thread(target=display_client_stream, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_frame', methods=["POST"])
def update_frame():
    """ Receives frames from the client, converts to grayscale, and stores them """
    global global_frame, original_frame
    data = request.json.get("frame")
    if data:
        header, encoded = data.split(',', 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Store the original frame before processing
        with frame_lock:
            original_frame = img.copy()

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels for streaming

        # Store processed black & white frame
        with frame_lock:
            global_frame = gray_img
    return '', 204

@app.route('/processed_feed')
def processed_feed():
    """ Streams the black & white processed client camera feed to the client """
    def generate():
        while True:
            with frame_lock:
                if global_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', global_frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # Reduce lag by optimizing streaming timing
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=(cert_path, key_path))
