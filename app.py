# app.py
from flask import Flask, render_template, Response, jsonify, request
import cv2
import easyocr
import pandas as pd
from datetime import datetime
import os
import threading
from queue import Queue
import time
import numpy as np

app = Flask(__name__)

# Global variables
frame_queue = Queue(maxsize=2)
detected_plate_queue = Queue(maxsize=2)
camera_active = False
reader = easyocr.Reader(['en'])
current_camera_url = "http://192.0.0.4:8080/video"
last_frame_time = time.time()
FRAME_INTERVAL = 1/30
last_captured_plate = None
last_capture_time = 0
CAPTURE_COOLDOWN = 3
class PlateDetector:
    def __init__(self):
        self.harcascade = "model/haarcascade_plate_number.xml"
        self.min_area = 500
        self.plate_cascade = cv2.CascadeClassifier(self.harcascade)
        self.process_every_n_frames = 2  # Reduced from 3 to 2
        self.frame_count = 0
        self.last_detected_plate = None
        self.last_detection_time = 0
        self.detection_cooldown = 0.5  # Reduced from 2 to 0.5 seconds
        self.last_valid_plate = None
        self.last_valid_coords = None
        self.plate_stability_counter = 0
        self.min_stable_frames = 3
    
    def detect_plate(self, img):
        if img is None:
            return None, None
        
        self.frame_count += 1
        processed_img = img.copy()
        current_time = time.time()
        
        # If we have a stable plate detection, keep showing it
        if self.last_valid_coords is not None:
            x, y, w, h = self.last_valid_coords
            cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(processed_img, "Number Plate", (x, y - 5),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            
            # Check if we should process a new frame
            if current_time - self.last_detection_time < self.detection_cooldown:
                return processed_img, self.last_valid_plate
        
        # Process new frame for detection
        if self.frame_count % self.process_every_n_frames == 0:
            img_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            scale_factor = 0.5
            small_img = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor)
            
            plates = self.plate_cascade.detectMultiScale(small_img,
                                                       scaleFactor=1.1,
                                                       minNeighbors=4,  # Reduced from 5 to 4
                                                       minSize=(int(50*scale_factor), int(15*scale_factor)))  # Added minSize
            
            plates = [(int(x/scale_factor), int(y/scale_factor),
                      int(w/scale_factor), int(h/scale_factor)) for x, y, w, h in plates]
            
            for (x, y, w, h) in plates:
                area = w * h
                if area > self.min_area:
                    # Check if this detection is close to the previous one
                    if self.last_valid_coords is not None:
                        prev_x, prev_y, _, _ = self.last_valid_coords
                        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                        if distance < 50:  # If detection is close to previous
                            self.plate_stability_counter += 1
                        else:
                            self.plate_stability_counter = 0
                    
                    cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(processed_img, "Number Plate", (x, y - 5),
                               cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                    
                    plate_roi = img[y:y+h, x:x+w]
                    self.last_detection_time = current_time
                    self.last_valid_coords = (x, y, w, h)
                    self.last_valid_plate = plate_roi
                    
                    # Only return the plate if we have a stable detection
                    if self.plate_stability_counter >= self.min_stable_frames:
                        return processed_img, plate_roi
                    return processed_img, None
        
        # If no detection in this frame but we have a recent valid detection
        if self.last_valid_coords is not None and current_time - self.last_detection_time > 1.0:
            self.last_valid_coords = None
            self.last_valid_plate = None
            self.plate_stability_counter = 0
            
        return processed_img, None

class CameraStream:
    def __init__(self, url):
        self.url = url
        self.stream = cv2.VideoCapture(self.url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        # Start frame grabbing thread
        self.grabbed = False
        self.frame = None
        self.read_thread = threading.Thread(target=self._read_frames)
        self.read_thread.daemon = True
        self.running = True
        self.read_thread.start()
        
    def _read_frames(self):
        while self.running:
            if self.stream.isOpened():
                self.grabbed, self.frame = self.stream.read()
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def read(self):
        return self.grabbed, self.frame
    
    def release(self):
        self.running = False
        if self.read_thread.is_alive():
            self.read_thread.join()
        self.stream.release()

def camera_stream():
    global camera_active, current_camera_url, last_frame_time
    detector = PlateDetector()
    
    # Initialize camera with threaded stream
    camera = CameraStream(current_camera_url)
    time.sleep(1)  # Allow camera to initialize
    
    connection_retry = 0
    max_retries = 5
    
    while camera_active:
        current_time = time.time()
        
        # Control frame rate
        if current_time - last_frame_time < FRAME_INTERVAL:
            continue
            
        success, img = camera.read()
        if not success:
            connection_retry += 1
            print(f"Failed to capture image. Attempt {connection_retry} of {max_retries}")
            
            if connection_retry >= max_retries:
                print("Camera connection failed. Stopping stream.")
                break
                
            # Try to reconnect
            camera.release()
            camera = CameraStream(current_camera_url)
            time.sleep(1)
            continue
            
        connection_retry = 0  # Reset retry counter on successful capture
        last_frame_time = current_time
        
        # Process frame
        img, plate_roi = detector.detect_plate(img)
        if plate_roi is not None and not detected_plate_queue.full():
            detected_plate_queue.put(plate_roi)
            
        # Compress frame before sending
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', img, encode_param)
        frame = buffer.tobytes()
        
        # Update frame queue
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()  # Remove old frame
                frame_queue.put(frame)    # Add new frame
            except:
                pass
    
    camera.release()

def save_to_database(plate_text, entry_type):
    global last_captured_plate, last_capture_time
    current_time = time.time()
    
    # Check if this is a duplicate capture within the cooldown period
    if (last_captured_plate == plate_text and 
        current_time - last_capture_time < CAPTURE_COOLDOWN):
        return None
    
    try:
        df = pd.read_excel("static/db.xlsx", engine="openpyxl")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Plate Number", "Date", "Time", "Entry Type", "ID"])
    
    # Update last captured plate info
    last_captured_plate = plate_text
    last_capture_time = current_time
    
    # Generate a unique ID for the entry
    new_id = str(int(time.time() * 1000))  # Using timestamp as ID
    
    new_entry = {
        "Plate Number": plate_text,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Entry Type": entry_type,
        "ID": new_id
    }
    
    # Convert existing IDs to strings to ensure consistency
    df['ID'] = df['ID'].astype(str)
    
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel("static/db.xlsx", index=False, engine="openpyxl")
    return new_entry
@app.route('/')
def index():
    return render_template('index.html', camera_url=current_camera_url)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)  # Prevent busy-waiting

@app.route('/update_camera_url', methods=['POST'])
def update_camera_url():
    global current_camera_url, camera_active
    data = request.get_json()
    new_url = data.get('url')
    
    if new_url:
        # Stop existing camera stream
        camera_active = False
        # Clear queues
        while not frame_queue.empty():
            frame_queue.get()
        while not detected_plate_queue.empty():
            detected_plate_queue.get()
            
        # Update URL and restart camera
        current_camera_url = new_url
        return jsonify({"status": "success", "message": "Camera URL updated"})
    return jsonify({"status": "error", "message": "Invalid URL"})

@app.route('/start_camera')
def start_camera():
    global camera_active
    if not camera_active:
        camera_active = True
        threading.Thread(target=camera_stream).start()
    return jsonify({"status": "success"})

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({"status": "success"})

@app.route('/capture_plate')
def capture_plate():
    entry_type = request.args.get('entry_type', 'IN')  # Get entry type from request
    
    # Clear the queue of old detections first
    while not detected_plate_queue.empty():
        _ = detected_plate_queue.get()
    
    # Wait briefly for a new detection
    time.sleep(0.5)
    
    if not detected_plate_queue.empty():
        plate_img = detected_plate_queue.get()
        cv2.imwrite("static/plates/scanned_img.jpg", plate_img)
        
        # Perform OCR
        results = reader.readtext(plate_img)
        plate_text = " ".join([res[1] for res in results])
        
        # Save to database with duplicate checking and entry type
        entry = save_to_database(plate_text, entry_type)
        if entry:
            return jsonify({"status": "success", "plate_text": plate_text, "entry": entry})
        else:
            return jsonify({"status": "error", "message": "Duplicate plate detected"})
    
    return jsonify({"status": "error", "message": "No plate detected"})

@app.route('/delete_entry', methods=['POST'])
def delete_entry():
    try:
        data = request.get_json()
        entry_id = data.get('id')
        
        # Read the database
        df = pd.read_excel("static/db.xlsx", engine="openpyxl")
        
        # Remove the entry with matching ID
        df = df[df['ID'] != entry_id]
        
        # Save the updated database
        df.to_excel("static/db.xlsx", index=False, engine="openpyxl")
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_entries')
def get_entries():
    try:
        df = pd.read_excel("static/db.xlsx", engine="openpyxl")
        # Convert DataFrame to list of dictionaries
        entries = df.to_dict('records')
        return jsonify({"status": "success", "entries": entries})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "entries": []})

if __name__ == '__main__':
    os.makedirs("static/plates", exist_ok=True)
    app.run(debug=True)