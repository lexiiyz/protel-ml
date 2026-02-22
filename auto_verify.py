import cv2
import os
import sys
import json
import tempfile
import dlib
import numpy as np
import face_recognition
import random
import time
import base64
import requests
import threading
from datetime import datetime
from scipy.spatial import distance as dist
from ultralytics import YOLO
import easyocr
import os
import time
import threading
import cv2
import numpy as np
import dlib
import face_recognition
import base64
import math

# YOLO Model Configuration
MODEL_PATH = "bestv2Revised.pt"  # Path to your local YOLO model
CONF_THRESHOLD = 0.4   # Confidence threshold for detections

# Load YOLO model once at startup
try:
    yolo_model = YOLO(MODEL_PATH)
    print("✅ Local YOLO model (bestv2.pt) loaded successfully")
    print("📋 Model classes:", yolo_model.names)
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    yolo_model = None

class AsyncOCRDetector:
    def __init__(self):
        self.is_detecting = False
        self.last_result = None
        self.detection_thread = None
        self.frame_queue = None
        # FRAME SAVING DISABLED - Frame capture attributes
        # self.capture_frames = False
        # self.timestamp_prefix = ""
        # self.step_frame_saved = False
        # self.step_name = ""
    
    def start_detection(self, frame, capture_frame=False, step_name=""):
        """Start asynchronous OCR detection if not already running"""
        if not self.is_detecting:
            self.is_detecting = True
            self.frame_queue = frame.copy()
            # FRAME SAVING DISABLED - Frame capture initialization
            # self.capture_frames = capture_frame
            # self.step_name = step_name
            # if capture_frame:
            #     self.timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.detection_thread = threading.Thread(target=self._detect_async)
            self.detection_thread.daemon = True
            self.detection_thread.start()
    
    def _detect_async(self):
        """Async OCR detection in background thread"""
        try:
            if self.frame_queue is not None:
                # Run OCR detection
                ocr_result = detect_vest_number_ocr(self.frame_queue)
                
                # Build result
                result = {
                    "success": ocr_result.get("success", False),
                    "vest_number": ocr_result.get("vest_number"),
                    "numbers_detected": ocr_result.get("numbers_detected", []),
                    "best_confidence": ocr_result.get("best_confidence", 0),
                    "total_detections": ocr_result.get("total_detections", 0),
                    "ocr_result": ocr_result
                }
                
                self.last_result = result
                
                # FRAME SAVING DISABLED - Save step frame if OCR detected numbers
                # if self.capture_frames and self.frame_queue is not None:
                #     self._save_ocr_step_frame_if_needed(result)
                    
        except Exception as e:
            print(f"⚠️ Async OCR detection error: {e}")
            self.last_result = {"success": False, "error": str(e)}
        finally:
            self.is_detecting = False
    
    def get_result(self):
        """Get latest OCR detection result (non-blocking)"""
        return self.last_result
    
class AsyncPPEDetector:
    def __init__(self):
        self.is_detecting = False
        self.last_result = None
        self.detection_thread = None
        self.frame_queue = None
    
    def start_detection(self, frame, capture_frame=False, step_name=""):
        """Start asynchronous PPE detection if not already running"""
        if not self.is_detecting:
            self.is_detecting = True
            self.frame_queue = frame.copy()
            self.detection_thread = threading.Thread(target=self._detect_async)
            self.detection_thread.daemon = True
            self.detection_thread.start()
    
    def _detect_async(self):
        """Async PPE detection in background thread - now uses local YOLO"""
        try:
            if self.frame_queue is not None:
                # Single YOLO detection call - no duplication
                ppe_predictions = detect_ppe_in_frame(self.frame_queue)
                person_result = detect_person_with_face_recognition(self.frame_queue)

                # Process results manually to avoid double YOLO call
                helmet_detected = False
                gloves_detected = False
                vest_detected = False
                boots_detected = False
                mask_detected = False
                glasses_detected = False

                for prediction in ppe_predictions.get('predictions', []):
                    class_name = prediction.get('class', '').lower()
                    confidence = prediction.get('confidence', 0)

                    if confidence > 0.5:  # Confidence threshold
                        # bestv2.pt class mapping: 0=helmet, 8=gloves, 3=vest, 9=boots
                        if class_name == '0':  # helmet
                            helmet_detected = True
                        elif class_name == '8':  # safety gloves
                            gloves_detected = True
                        elif class_name == '3':  # safety vest
                            vest_detected = True
                        elif class_name == '9':  # safety boots
                            boots_detected = True
                        elif class_name == '4':  # safety mask
                            mask_detected = True
                        elif class_name == '6':  # safety glasses
                            glasses_detected = True

                # Build comprehensive result
                result = {
                    "success": True,
                    "helmet_detected": helmet_detected,
                    "gloves_detected": gloves_detected,
                    "vest_detected": vest_detected,
                    "boots_detected": boots_detected,
                    "mask_detected": mask_detected,
                    "glasses_detected": glasses_detected,
                    "person_detected": person_result.get("person_detected", False),
                    "ppe_predictions": ppe_predictions
                }

                self.last_result = result
                
        except Exception as e:
            print(f"⚠️ Async PPE detection error: {e}")
            self.last_result = {"success": False, "error": str(e)}
        finally:
            self.is_detecting = False
    
    def get_result(self):
        """Get latest detection result (non-blocking)"""
        return self.last_result
    
class LivenessDetector:
    def __init__(self):
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        
        # Try to load the shape predictor
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
            print("✅ Facial landmark predictor loaded successfully")
        else:
            print("❌ shape_predictor_68_face_landmarks.dat not found")
            print("Please ensure the file is in the same directory")
        
        # Eye aspect ratio threshold for blink detection
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 3
        
        # Counters for blink detection
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_timestamps = []
        self.current_challenge = None
        
        # Frame storage during liveness detection
        self.liveness_frames = []  # Store ALL frames during liveness period
        self.max_liveness_frames = 900  # Store up to 30 seconds at 30fps
    
    def eye_aspect_ratio(self, eye):
        """Calculate the eye aspect ratio (EAR)"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def analyze_face_pose_quality(self, landmarks, frame):
        """Analyze face pose and quality metrics"""
        if landmarks is None or len(landmarks) < 68:
            return 0, False, 0
        
        try:
            # Calculate face pose using key landmarks
            nose_tip = landmarks[30]
            left_eye_center = np.mean(landmarks[36:42], axis=0)
            right_eye_center = np.mean(landmarks[42:48], axis=0)
            chin = landmarks[8]
            
            # Check if face is frontal (eyes should be roughly horizontal)
            eye_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], 
                                 right_eye_center[0] - left_eye_center[0])
            is_frontal = abs(np.degrees(eye_angle)) < 12  # Within 12 degrees
            
            # Calculate face symmetry
            face_left = landmarks[0]
            face_right = landmarks[16]
            face_center_x = (face_left[0] + face_right[0]) / 2
            nose_offset = abs(nose_tip[0] - face_center_x)
            face_width = face_right[0] - face_left[0]
            symmetry_ratio = max(0, 1 - (nose_offset / face_width)) if face_width > 0 else 0
            
            # Calculate pose stability score
            eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
            face_height = np.linalg.norm(nose_tip - chin)
            pose_stability = min(1.0, eye_distance / max(1, face_height * 0.3))
            
            return symmetry_ratio, is_frontal, pose_stability
            
        except Exception:
            return 0, False, 0
    
    def calculate_comprehensive_quality(self, frame, landmarks, ear_value, eyes_open):
        """Calculate comprehensive frame quality during liveness detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness analysis
            brightness = np.mean(gray)
            brightness_score = max(0, 100 - abs(brightness - 128))  # Prefer balanced lighting
            
            # Face pose and stability
            symmetry, is_frontal, pose_stability = self.analyze_face_pose_quality(landmarks, frame)
            
            # Eyes open bonus (prefer eyes-open frames for final recognition)
            eyes_bonus = 100 if eyes_open else 0
            
            # EAR stability (prefer stable eye states)
            ear_stability = max(0, 100 - abs(ear_value - 0.3) * 200)  # Prefer normal open eyes
            
            # Combine all quality metrics
            quality_score = (
                laplacian_var * 0.3 +           # 30% sharpness
                brightness_score * 0.2 +        # 20% lighting
                symmetry * 100 * 0.2 +          # 20% face symmetry
                pose_stability * 100 * 0.1 +    # 10% pose stability
                eyes_bonus * 0.15 +             # 15% eyes open preference
                ear_stability * 0.05            # 5% ear stability
            )
            
            return quality_score, is_frontal
            
        except Exception as e:
            return 0, False
    
    def store_liveness_frame(self, frame, landmarks, ear_value, eyes_open, blink_phase):
        """Store frame during liveness detection with comprehensive metadata"""
        try:
            # Calculate quality metrics
            quality_score, is_frontal = self.calculate_comprehensive_quality(
                frame, landmarks, ear_value, eyes_open)
            
            # Create frame record
            frame_record = {
                "frame": frame.copy(),
                "timestamp": time.time(),
                "quality_score": quality_score,
                "ear_value": ear_value,
                "eyes_open": eyes_open,
                "is_frontal": is_frontal,
                "blink_phase": blink_phase,  # "open", "closing", "closed", "opening"
                "blink_count_at_capture": self.total_blinks,
                "frame_index": len(self.liveness_frames)
            }
            
            # Store frame (manage memory by limiting storage)
            self.liveness_frames.append(frame_record)
            
            # Remove oldest frames if exceeding limit
            if len(self.liveness_frames) > self.max_liveness_frames:
                self.liveness_frames.pop(0)
                
        except Exception as e:
            print(f"⚠️ Error storing liveness frame: {e}")
    
    def detect_blink_with_storage(self, frame):
        """Detect blinks and store all frames during liveness period"""
        if self.predictor is None:
            return False, frame, 0, True, "no_predictor"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        blink_detected = False
        current_ear = 0
        eyes_open = True
        blink_phase = "unknown"
        landmarks = None
        
        for face in faces:
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            
            # Extract eye coordinates
            left_eye = landmarks_array[36:42]
            right_eye = landmarks_array[42:48]
            
            # Calculate EAR for both eyes
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            current_ear = (left_ear + right_ear) / 2.0
            
            # Determine eye state and blink phase
            eyes_open = current_ear >= self.EAR_THRESHOLD
            
            if current_ear < self.EAR_THRESHOLD:
                if self.blink_counter == 0:
                    blink_phase = "closing"
                else:
                    blink_phase = "closed"
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.CONSECUTIVE_FRAMES:
                    self.total_blinks += 1
                    self.blink_timestamps.append(time.time())
                    blink_detected = True
                    blink_phase = "opening"
                    print(f"👁️ Blink {self.total_blinks} detected during liveness period")
                else:
                    blink_phase = "open"
                self.blink_counter = 0
            
            # Store this frame with metadata
            self.store_liveness_frame(frame, landmarks_array, current_ear, eyes_open, blink_phase)
            
            break  # Only process first detected face
        
        return blink_detected, frame, current_ear, eyes_open, blink_phase
    
    def generate_blink_challenge(self):
        """Generate blink challenge - defaults to 3 blinks for consistency"""
        self.current_challenge = {"blinks": 3, "text": "Blink 3 times"}
        self.total_blinks = 0
        self.blink_timestamps = []
        self.liveness_frames = []  # Reset frame storage
        print("🎯 Challenge: Blink 3 times for liveness verification")
        print("📸 All frames during challenge will be stored and analyzed")
        return self.current_challenge
    
    def validate_human_timing(self):
        """Validate blink timing is human-like"""
        if len(self.blink_timestamps) < 2:
            return True
        
        intervals = []
        for i in range(len(self.blink_timestamps) - 1):
            interval = self.blink_timestamps[i + 1] - self.blink_timestamps[i]
            intervals.append(interval)
        
        # Human blinks: reasonable intervals and natural variation
        valid_intervals = all(0.2 <= interval <= 3.0 for interval in intervals)
        natural_timing = True
        if len(intervals) > 2:
            timing_variance = np.std(intervals)
            natural_timing = timing_variance > 0.1
        
        return valid_intervals and natural_timing
    
    def select_best_frame_from_liveness_period(self):
        """Select the absolute best frame from the liveness detection period"""
        if not self.liveness_frames:
            print("❌ No frames stored during liveness period")
            return None, {}
        
        print(f"🔍 Analyzing {len(self.liveness_frames)} frames from liveness period...")
        
        # Filter for high-quality eyes-open frontal frames
        quality_frames = []
        for frame_record in self.liveness_frames:
            if (frame_record["eyes_open"] and 
                frame_record["is_frontal"] and 
                frame_record["quality_score"] > 100):  # Minimum quality threshold
                quality_frames.append(frame_record)
        
        if not quality_frames:
            print("⚠️ No high-quality eyes-open frontal frames found, using best available")
            quality_frames = [max(self.liveness_frames, key=lambda x: x["quality_score"])]
        
        # Sort by quality score and select the best
        quality_frames.sort(key=lambda x: x["quality_score"], reverse=True)
        best_frame_record = quality_frames[0]
        
        # Create selection report
        selection_report = {
            "total_frames_analyzed": len(self.liveness_frames),
            "quality_frames_found": len(quality_frames),
            "selected_frame_quality": best_frame_record["quality_score"],
            "selected_frame_ear": best_frame_record["ear_value"],
            "selected_frame_phase": best_frame_record["blink_phase"],
            "selected_frame_timestamp": best_frame_record["timestamp"],
            "selection_method": "highest_quality_eyes_open_frontal"
        }
        
        print(f"✅ Best frame selected: Quality={best_frame_record['quality_score']:.1f}, "
              f"EAR={best_frame_record['ear_value']:.3f}, Phase={best_frame_record['blink_phase']}")
        
        return best_frame_record["frame"], selection_report

def load_reference_encodings(reference_path):
    """Load and encode the reference image"""
    if not os.path.exists(reference_path):
        print(f"❌ Reference image not found: {reference_path}")
        return None
    
    try:
        reference_image = face_recognition.load_image_file(reference_path)
        reference_encodings = face_recognition.face_encodings(reference_image)
        
        if len(reference_encodings) == 0:
            print(f"❌ No face found in reference image: {reference_path}")
            return None
        
        print(f"✅ Reference face loaded successfully")
        return reference_encodings[0]
        
    except Exception as e:
        print(f"❌ Error loading reference image: {e}")
        return None

def load_reference_encodings_multi(reference_dir="references", fallback_file="reference.jpg"):
    """
    Load all reference images in reference_dir and return dict {filename: encoding}.
    If none found, try fallback_file.
    """
    encodings = {}
    try:
        if os.path.isdir(reference_dir):
            for fname in sorted(os.listdir(reference_dir)):
                path = os.path.join(reference_dir, fname)
                if not os.path.isfile(path):
                    continue
                try:
                    img = face_recognition.load_image_file(path)
                    enc = face_recognition.face_encodings(img)
                    if len(enc) > 0:
                        encodings[fname] = enc[0]
                        print(f"✅ Loaded reference: {fname}")
                    else:
                        print(f"⚠️ No face found in reference file: {fname}")
                except Exception as e:
                    print(f"⚠️ Failed to load {fname}: {e}")
        # fallback to single reference file
        if not encodings and os.path.exists(fallback_file):
            enc = load_reference_encodings(fallback_file)
            if enc is not None:
                encodings[os.path.basename(fallback_file)] = enc
    except Exception as e:
        print(f"❌ Error loading reference encodings: {e}")
    return encodings

def verify_face_against_references(rgb_image, reference_encodings_dict, threshold=50.0):
    """
    Compare rgb_image against all reference encodings.
    Returns dict with best match and distances.
    """
    try:
        if not reference_encodings_dict:
            return {"success": False, "error": "No reference encodings", "match": False, "match_percent": 0, "matched_name": None}
        face_encs = face_recognition.face_encodings(rgb_image)
        if len(face_encs) == 0:
            return {"success": False, "error": "No face found", "match": False, "match_percent": 0, "matched_name": None}
        probe = face_encs[0]
        best_name = None
        best_percent = 0.0
        distances = {}
        for name, ref_enc in reference_encodings_dict.items():
            d = face_recognition.face_distance([ref_enc], probe)[0]
            percent = max(0.0, (1.0 - d) * 100.0)
            distances[name] = round(percent, 2)
            if percent > best_percent:
                best_percent = percent
                best_name = name
        match = best_percent >= threshold
        return {"success": True, "match": match, "match_percent": round(best_percent,1), "matched_name": best_name, "distances": distances}
    except Exception as e:
        return {"success": False, "error": str(e), "match": False, "match_percent": 0, "matched_name": None}

def verify_face_direct(frame, reference_encoding, threshold=50.0):
    """Direct face verification using selected frame"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        
        if len(face_encodings) == 0:
            return {"success": False, "error": "No face found in frame", "match": False, "match_percent": 0}
        
        # Compare with reference
        face_distances = face_recognition.face_distance([reference_encoding], face_encodings[0])
        distance = face_distances[0]
        match_percent = max(0, (1 - distance) * 100)
        match = match_percent >= threshold
        
        return {
            "success": True,
            "match": match,
            "match_percent": round(match_percent, 1),
            "threshold_used": threshold
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "match": False, "match_percent": 0}

def encode_image_to_base64(frame):
    """Convert frame to base64 for API request (from ppe_detect.py)"""
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def detect_ppe_yolo_local(frame, conf=CONF_THRESHOLD):
    """
    Run the local YOLO model on a BGR OpenCV frame.
    Returns: dict compatible with Roboflow format
    """
    if yolo_model is None:
        return {"predictions": []}
    
    try:
        # Run YOLO prediction
        results = yolo_model.predict(source=frame, imgsz=640, conf=conf, verbose=False, device='cpu')
        
        predictions = []
        if len(results) > 0:
            r = results[0]
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Convert to Roboflow format (center x, center y, width, height)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    class_name = yolo_model.names.get(cls, str(cls)).lower()
                    
                    # Debug output
                    print(f"DEBUG - YOLO Detection: {class_name} = {confidence:.3f}")
                    
                    predictions.append({
                        "class": class_name,
                        "confidence": confidence,
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height
                    })
        
        return {"predictions": predictions}
        
    except Exception as e:
        print(f"❌ YOLO detection error: {e}")
        return {"predictions": []}

def detect_ppe_in_frame(frame):
    if yolo_model is None: return {"predictions": []}
    
    try:
        results = yolo_model.predict(source=frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False, device='cpu')
        predictions = []
        
        if len(results) > 0:
            r = results[0]
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    class_name = yolo_model.names.get(cls_id, str(cls_id)).lower()
                    
                    # --- AMBIL KOORDINAT BOX ---
                    # xywhn = x center, y center, width, height (normalized 0-1)
                    # Kita pakai normalized agar mudah disesuaikan di frontend berapapun ukuran layarnya
                    x, y, w, h = box.xywhn[0].tolist()
                    
                    predictions.append({
                        "class": class_name,
                        "confidence": conf,
                        "box": {"x": x, "y": y, "w": w, "h": h} # Kirim koordinat
                    })
                    
                    sys.stderr.write(f"   👉 {class_name}: {conf:.2f} [x={x:.2f}, y={y:.2f}]\n")

        return {"predictions": predictions}

    except Exception as e:
        sys.stderr.write(f"❌ YOLO Error: {e}\n")
        return {"predictions": []}

def align_face_by_eyes(image, left_eye, right_eye):
    """Rotate image so eyes are horizontal"""
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    if dx == 0 and dy == 0:
        return image
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_face_crop(frame, bbox=None, landmarks=None, size=160, padding=0.25, do_align=True, save_stages=False, stage_prefix=""):
    """
    Crop & preprocess face for encoding with optional stage saving.
    - frame: BGR image (full frame)
    - bbox: (x1,y1,x2,y2) optional; if None will try to compute from landmarks
    - landmarks: optional 68x2 numpy array in frame coords (preferred for alignment)
    - save_stages: if True, saves each preprocessing stage to Pre-Process folder
    - stage_prefix: prefix for saved stage files
    - returns: RGB uint8 image sized (size,size) or None on failure
    """
    h, w = frame.shape[:2]
    
    # FRAME SAVING DISABLED - Create Pre-Process directory if it doesn't exist
    # if save_stages:
    #     os.makedirs("Pre-Process", exist_ok=True)

    # If landmarks provided but bbox not, estimate bbox from landmarks
    if landmarks is not None and bbox is None:
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        x1 = int(max(0, xs.min() - (xs.max()-xs.min()) * 0.25))
        y1 = int(max(0, ys.min() - (ys.max()-ys.min()) * 0.35))
        x2 = int(min(w, xs.max() + (xs.max()-xs.min()) * 0.25))
        y2 = int(min(h, ys.max() + (ys.max()-ys.min()) * 0.35))
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
    else:
        return None

    # Apply padding
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)
    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(w, x2 + pad_w)
    y2p = min(h, y2 + pad_h)

    crop = frame[y1p:y2p, x1p:x2p].copy()
    if crop.size == 0:
        return None
    
    # FRAME SAVING DISABLED - Stage 1: Original crop
    # if save_stages:
    #     cv2.imwrite(f"Pre-Process/{stage_prefix}1_original_crop.jpg", crop)

    # Stage 2: Align by eyes if landmarks available
    if do_align and landmarks is not None:
        try:
            left_eye = np.mean(landmarks[36:42], axis=0).astype(int)
            right_eye = np.mean(landmarks[42:48], axis=0).astype(int)
            # convert to crop coords
            left_eye_c = (left_eye[0] - x1p, left_eye[1] - y1p)
            right_eye_c = (right_eye[0] - x1p, right_eye[1] - y1p)
            crop = align_face_by_eyes(crop, left_eye_c, right_eye_c)
            
            # FRAME SAVING DISABLED - Stage 2: Aligned
            # if save_stages:
            #     cv2.imwrite(f"Pre-Process/{stage_prefix}2_aligned.jpg", crop)
        except Exception:
            pass

    # Stage 3: CLAHE (improve contrast)
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # FRAME SAVING DISABLED - Stage 3: CLAHE
        # if save_stages:
        #     cv2.imwrite(f"Pre-Process/{stage_prefix}3_clahe.jpg", crop)
    except Exception:
        pass

    # Stage 4: Denoise (mild)
    try:
        crop = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)
        
        # FRAME SAVING DISABLED - Stage 4: Denoised
        # if save_stages:
        #     cv2.imwrite(f"Pre-Process/{stage_prefix}4_denoised.jpg", crop)
    except Exception:
        pass

    # Stage 5: Unsharp mask (mild sharpen)
    try:
        gaussian = cv2.GaussianBlur(crop, (0,0), 3)
        crop = cv2.addWeighted(crop, 1.2, gaussian, -0.2, 0)
        
        # FRAME SAVING DISABLED - Stage 5: Sharpened
        # if save_stages:
        #     cv2.imwrite(f"Pre-Process/{stage_prefix}5_sharpened.jpg", crop)
    except Exception:
        pass

    # Stage 6: Resize to consistent size
    try:
        crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        
        # FRAME SAVING DISABLED - Stage 6: Resized
        # if save_stages:
        #     cv2.imwrite(f"Pre-Process/{stage_prefix}6_resized.jpg", crop_resized)
    except Exception:
        return None

    # Stage 7: Convert to RGB (final stage)
    rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    
    # FRAME SAVING DISABLED - Stage 7: Final RGB
    # if save_stages:
    #     # Save RGB as BGR for viewing
    #     bgr_final = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f"Pre-Process/{stage_prefix}7_final_rgb.jpg", bgr_final)
    
    return rgb

def verify_face_with_preprocessed(rgb_face, reference_encoding, threshold=50.0):
    """
    Verify a preprocessed RGB face crop against reference encoding.
    Returns same structure as verify_face_direct.
    """
    try:
        # face_recognition expects RGB numpy array
        face_encodings = face_recognition.face_encodings(rgb_face)
        if len(face_encodings) == 0:
            return {"success": False, "error": "No face found in preprocessed crop", "match": False, "match_percent": 0}

        distance = face_recognition.face_distance([reference_encoding], face_encodings[0])[0]
        match_percent = max(0, (1 - distance) * 100)
        match = match_percent >= threshold
        return {"success": True, "match": match, "match_percent": round(match_percent,1), "threshold_used": threshold}
    except Exception as e:
        return {"success": False, "error": str(e), "match": False, "match_percent": 0}

def detect_person_with_face_recognition(frame):
    """Detect person using face_recognition (from auto_verify-backup.py approach)"""
    try:
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use face_recognition to detect faces (indicates person present)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # If faces found, person is detected
        person_detected = len(face_locations) > 0
        
        return {
            "success": True,
            "person_detected": person_detected,
            "face_count": len(face_locations),
            "face_locations": face_locations
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "person_detected": False}

def draw_predictions(frame, predictions):
    """Draw bounding boxes on frame (from ppe_detect.py)"""
    if 'predictions' not in predictions:
        return frame
    
    for prediction in predictions['predictions']:
        # Get bounding box coordinates
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        
        # Calculate box corners
        x1 = int(x - width/2)
        y1 = int(y - height/2)
        x2 = int(x + width/2)
        y2 = int(y + height/2)
        
        # Get class name and confidence
        class_name = prediction['class']
        confidence = prediction['confidence']
        
        # Draw rectangle - RED for helmet (bad in step 1), GREEN for other PPE
        color = (0, 0, 255) if class_name.lower() == 'hardhat' else (0, 255, 0)  # RED for hardhat, GREEN for others
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def detect_helmet_status(frame):
    """Detect helmet presence using EXACT methods from ppe_detect.py + face detection for person"""
    try:
        # Get PPE detections
        ppe_result = detect_ppe_in_frame(frame)
        
        # Get person detection via face recognition
        person_result = detect_person_with_face_recognition(frame)
        
        # Process PPE results
        helmet_detected = False
        ppe_person_detected = False
        
        for prediction in ppe_result.get('predictions', []):
            class_name = prediction.get('class', '').lower()
            confidence = prediction.get('confidence', 0)
            
            print(f"DEBUG - PPE Detection: {class_name} = {confidence:.3f}")
            
            # Check for helmet - bestv2.pt uses class '0' for helmet
            if class_name == '0' and confidence > 0.4:  # 40% threshold
                helmet_detected = True
                print(f"DEBUG - HELMET DETECTED: {class_name} = {confidence:.3f}")
            # Note: PPE model may not detect "person" - it only detects PPE items
            # We'll rely on face_recognition for person detection
        
        # Use face detection as primary person detection
        face_person_detected = person_result.get("person_detected", False)
        
        # Person detected if either PPE API detects person OR face is detected
        person_detected = ppe_person_detected or face_person_detected
        
        print(f"DEBUG - FINAL RESULT: helmet={helmet_detected}, ppe_person={ppe_person_detected}, face_person={face_person_detected}, final_person={person_detected}")
        
        return {
            "success": True,
            "helmet_detected": helmet_detected,
            "person_detected": person_detected,
            "face_detected": face_person_detected,
            "ppe_person_detected": ppe_person_detected,
            "raw_ppe_result": ppe_result,
            "raw_person_result": person_result
        }
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return {"success": False, "error": str(e)}

def check_no_helmet_requirement(frame):
    """Check that user is NOT wearing helmet"""
    result = detect_helmet_status(frame)
    
    if not result["success"]:
        print(f"DEBUG - PPE Detection failed: {result.get('error', 'Detection failed')}")
        return {"passed": False, "error": result.get("error", "Detection failed")}
    
    # For step 1, we want NO helmet detected
    no_helmet = not result["helmet_detected"]
    person_present = result["person_detected"]
    face_detected = result.get("face_detected", False)
    ppe_person = result.get("ppe_person_detected", False)
    
    print(f"DEBUG - DETAILED PPE Result:")
    print(f"   Helmet detected: {result['helmet_detected']}")
    print(f"   Person via PPE: {ppe_person}")
    print(f"   Person via Face: {face_detected}")
    print(f"   Final person: {person_present}")
    print(f"   No helmet: {no_helmet}")
    print(f"   PASSED: {no_helmet and person_present}")
    
    return {
        "passed": no_helmet and person_present,
        "helmet_detected": result["helmet_detected"],
        "person_detected": person_present,
        "face_detected": face_detected,
        "ppe_person_detected": ppe_person,
        "message": "No helmet detected - good!" if no_helmet else "Please remove helmet to continue"
    }

def check_helmet_requirement(frame):
    """Check that user IS wearing helmet (NO person detection required)"""
    result = detect_helmet_status(frame)
    
    if not result["success"]:
        return {"passed": False, "error": result.get("error", "Detection failed")}
    
    # For PPE step, we ONLY want helmet detected (no person requirement)
    helmet_present = result["helmet_detected"]
    person_present = result["person_detected"]
    face_detected = result.get("face_detected", False)
    ppe_person = result.get("ppe_person_detected", False)
    
    print(f"DEBUG - PPE REQUIREMENT CHECK:")
    print(f"   Helmet detected: {helmet_present}")
    print(f"   Person via PPE: {ppe_person}")
    print(f"   Person via Face: {face_detected}")
    print(f"   Final person: {person_present}")
    print(f"   PASSED: {helmet_present} (person detection DISABLED for Step 5)")
    
    return {
        "passed": helmet_present,  # ONLY helmet required, no person check
        "helmet_detected": helmet_present,
        "person_detected": person_present,
        "face_detected": face_detected,
        "ppe_person_detected": ppe_person,
        "message": "Helmet detected - excellent!" if helmet_present else "Please wear helmet for PPE verification"
    }

def detect_gloves_status(frame):
    """Detect safety gloves using local YOLO model"""
    try:
        predictions = detect_ppe_yolo_local(frame, conf=CONF_THRESHOLD)
        
        gloves_detected = False
        person_detected = False
        
        for pred in predictions["predictions"]:
            class_name = pred["class"].lower()
            confidence = pred["confidence"]
            
            print(f"DEBUG - Gloves detection: {class_name} = {confidence:.3f}")
            
            if class_name == "8" and confidence >= CONF_THRESHOLD:  # bestv2.pt: 8 = safety gloves
                gloves_detected = True
                print(f"✅ Safety gloves detected with confidence: {confidence:.3f}")
            elif class_name == "person" and confidence >= CONF_THRESHOLD:
                person_detected = True
                print(f"✅ Person detected with confidence: {confidence:.3f}")
        
        return {
            "success": True,
            "gloves_detected": gloves_detected,
            "person_detected": person_detected,
            "ppe_predictions": predictions
        }
        
    except Exception as e:
        print(f"❌ Gloves detection error: {e}")
        return {
            "success": False,
            "error": str(e),
            "gloves_detected": False,
            "person_detected": False,
            "ppe_predictions": {"predictions": []}
        }

def detect_vest_number_ocr(frame):
    """Detect numbers on safety vest using EasyOCR"""
    try:
        # Initialize EasyOCR reader (cache it globally if needed)
        if not hasattr(detect_vest_number_ocr, 'reader'):
            print("🔍 Initializing EasyOCR reader for vest number detection...")
            try:
                detect_vest_number_ocr.reader = easyocr.Reader(['en'], gpu=False)  # Use CPU to avoid conflicts
                print("✅ EasyOCR reader initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize EasyOCR: {e}")
                return {"success": False, "error": str(e), "numbers_detected": [], "vest_number": None}
        
        # Run OCR on the frame to detect numbers
        results = detect_vest_number_ocr.reader.readtext(
            frame, 
            allowlist='0123456789',  # Only look for numeric digits
            width_ths=0.7,          # Text width threshold
            height_ths=0.7          # Text height threshold
        )
        
        detected_numbers = []
        best_number = None
        best_confidence = 0
        
        if results:
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Minimum confidence threshold
                    detected_numbers.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox
                    })
                    
                    # Track the best (highest confidence) number
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_number = text
                    
                    print(f"📋 OCR detected: '{text}' (confidence: {confidence:.3f})")
        
        return {
            "success": True,
            "numbers_detected": detected_numbers,
            "vest_number": best_number,
            "best_confidence": best_confidence,
            "total_detections": len(detected_numbers)
        }
        
    except Exception as e:
        print(f"❌ OCR vest number detection error: {e}")
        return {
            "success": False,
            "error": str(e),
            "numbers_detected": [],
            "vest_number": None
        }

def draw_ocr_results(frame, ocr_result):
    """Draw OCR detection results on frame"""
    try:
        if not ocr_result.get("success", False):
            return frame
        
        annotated_frame = frame.copy()
        numbers_detected = ocr_result.get("numbers_detected", [])
        
        for detection in numbers_detected:
            bbox = detection["bbox"]
            text = detection["text"]
            confidence = detection["confidence"]
            
            # Convert bbox coordinates to integers
            (p1, p2, p3, p4) = [(int(point[0]), int(point[1])) for point in bbox]
            
            # Draw bounding box (green for high confidence, yellow for medium)
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(annotated_frame, p1, p3, color, 2)
            
            # Draw text with confidence
            display_text = f"{text} ({confidence*100:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Black background for text readability
            (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)
            cv2.rectangle(annotated_frame, (p1[0], p1[1] - text_h - 10), (p1[0] + text_w, p1[1]), (0, 0, 0), -1)
            
            # White text
            cv2.putText(annotated_frame, display_text, (p1[0], p1[1] - 5), 
                       font, font_scale, (255, 255, 255), font_thickness)
        
        return annotated_frame
        
    except Exception as e:
        print(f"⚠️ Error drawing OCR results: {e}")
        return frame

def detect_vest_status(frame):
    """Detect safety vest using local YOLO model"""
    try:
        predictions = detect_ppe_yolo_local(frame, conf=CONF_THRESHOLD)
        
        vest_detected = False
        person_detected = False
        
        for pred in predictions["predictions"]:
            class_name = pred["class"].lower()
            confidence = pred["confidence"]
            
            print(f"DEBUG - Vest detection: {class_name} = {confidence:.3f}")
            
            if class_name == "3" and confidence >= CONF_THRESHOLD:  # bestv2.pt: 3 = safety vest
                vest_detected = True
                print(f"✅ Safety vest detected with confidence: {confidence:.3f}")
            elif class_name == "person" and confidence >= CONF_THRESHOLD:
                person_detected = True
                print(f"✅ Person detected with confidence: {confidence:.3f}")
        
        return {
            "success": True,
            "vest_detected": vest_detected,
            "person_detected": person_detected,
            "ppe_predictions": predictions
        }
        
    except Exception as e:
        print(f"❌ Vest detection error: {e}")
        return {
            "success": False,
            "error": str(e),
            "vest_detected": False,
            "person_detected": False,
            "ppe_predictions": {"predictions": []}
        }

def liveness_with_frame_selection_verification(reference_path="reference.jpg", threshold=50.0):
    """Advanced liveness verification with retrospective frame selection and PPE detection"""
    
    print("=== ENHANCED LIVENESS VERIFICATION WITH LOCAL PPE DETECTION ===")
    print("📋 Step 1: No-Helmet Check + Face Positioning (Local YOLO)")
    print("📋 Step 2: Liveness Challenge (Record ALL frames)")
    print("📋 Step 3: Select Best Frame from Liveness Period")
    print("📋 Step 4: Face Recognition with Selected Frame")
    print("📋 Step 5: PPE Helmet Detection (Local YOLO)")
    print("📋 Step 6: Safety Gloves Detection (Raise hands)")
    print("📋 Step 7: Safety Vest Detection (Step back)")
    print("📋 Step 8: Safety Boots Detection (Show boots)")
    print("\n🚀 Starting enhanced verification process...")
    print("⚠️ Step 1: Please ensure you are NOT wearing a helmet")
    print("📸 Position your face properly and remove any helmet")
    print("🎯 Then record all frames during liveness challenge")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return None, False, False, 0, False
    
    # Optimize webcam settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Initialize components
    liveness = LivenessDetector()
    use_liveness = liveness.predictor is not None
    
    # Load multiple reference encodings (references/ folder) with fallback to reference.jpg
    reference_encodings = load_reference_encodings_multi("references", fallback_file=reference_path)
    if not reference_encodings:
        print("❌ No reference encodings found (check 'references/' folder or reference.jpg)")
        cap.release()
        return None, False, False, 0, False, False, False, False, None
    
    # State variables
    current_step = 0  # Start with no-helmet + face positioning step
    final_frame = None
    final_match_percent = 0
    timing_valid = True
    liveness_completed = False
    selection_report = {}
    face_positioned = False
    blink_challenge = None
    no_helmet_verified = False
    face_recognition_success = False
    ppe_completed = False
    gloves_completed = False
    vest_completed = False
    boots_completed = False
    glasses_completed = False
    mask_completed = False
    matched_reference_name = None

    # Start with no-helmet + face positioning step
    if use_liveness:
        print(f"\n🎯 Step 1/8: Remove helmet and position your face...")
        print("⚠️ Please ensure you are NOT wearing a helmet")
        print("📸 Looking for a clear, frontal face WITHOUT helmet")
    else:
        print("⚠️ No liveness detection available")
        current_step = 3  # Skip to recognition

    try:
        frame_count = 0
        window_name = 'Liveness with Frame Selection Verification'
        
        # Main verification loop
        while current_step <= 10:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame")
                break

            frame_count += 1
            display_frame = frame.copy()

            # Add progress indicator
            total_steps = 10 if use_liveness else 3
            current_display_step = current_step if current_step > 0 else 1
            cv2.putText(display_frame, f"Step {current_display_step}/{total_steps}", (display_frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)            # STEP 0: No-Helmet Check + Face Positioning
            if current_step == 0:
                # Initialize positioning counter and PPE detection
                if not hasattr(liveness, 'positioning_frames'):
                    liveness.positioning_frames = 0
                if not hasattr(liveness, 'face_positioned'):
                    liveness.face_positioned = False
                if not hasattr(liveness, 'step1_async_ppe'):
                    liveness.step1_async_ppe = AsyncPPEDetector()
                    # FRAME SAVING DISABLED - Reset for new session
                    # liveness.step1_async_ppe.reset_saved_items()  # Reset for new session
                if not hasattr(liveness, 'step1_frame_count'):
                    liveness.step1_frame_count = 0
                if not hasattr(liveness, 'last_no_helmet_result'):
                    liveness.last_no_helmet_result = {"helmet_detected": True, "person_detected": False}  # Start assuming helmet present
                
                if use_liveness:
                    cv2.putText(display_frame, "STEP 1: Remove helmet & position face", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Ensure NO helmet, look directly at camera", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    
                    liveness.step1_frame_count += 1
                    
                    # Start async PPE detection every 30 frames for no-helmet check
                    if liveness.step1_frame_count % 30 == 0:
                        print(f"🔍 Starting async no-helmet check #{liveness.step1_frame_count // 30}...")
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.step1_async_ppe.start_detection(frame)  # capture_frame=True, step_name="No_Helmet_Check"
                    
                    # Check for async PPE detection results (non-blocking)
                    async_ppe_result = liveness.step1_async_ppe.get_result()
                    if async_ppe_result is not None:
                        if async_ppe_result.get("success", False):
                            helmet_detected = async_ppe_result.get("helmet_detected", True)
                            person_detected = async_ppe_result.get("person_detected", False)
                            
                            # Update last result and store predictions for drawing
                            liveness.last_no_helmet_result = {
                                "helmet_detected": helmet_detected,
                                "person_detected": person_detected,
                                "ppe_predictions": async_ppe_result.get("ppe_predictions", {"predictions": []})
                            }
                            
                            if helmet_detected:
                                print(f"❌ Helmet detected - please remove helmet")
                            else:
                                print(f"✅ No helmet detected - good!")
                        else:
                            print(f"⚠️ PPE detection error: {async_ppe_result.get('error', 'Unknown')}")
                    
                    # Draw PPE bounding boxes if available
                    if hasattr(liveness, 'last_no_helmet_result') and 'ppe_predictions' in liveness.last_no_helmet_result:
                        display_frame = draw_predictions(display_frame, liveness.last_no_helmet_result['ppe_predictions'])
                    
                    # Show PPE status
                    helmet_present = liveness.last_no_helmet_result.get("helmet_detected", True)
                    mask_present = liveness.last_no_helmet_result.get("mask_detected", False)
                    glasses_present = liveness.last_no_helmet_result.get("glasses_detected", False)
                    person_present = liveness.last_no_helmet_result.get("person_detected", False)
                    no_ppe_ok = not helmet_present and not mask_present and not glasses_present and person_present
                    
                    if helmet_present:
                        cv2.putText(display_frame, "❌ Please remove helmet", (10, 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    elif not person_present:
                        cv2.putText(display_frame, "❌ Move closer - no person detected", (10, 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "✅ No helmet detected", (10, 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show async detection status
                    detection_status = "Processing..." if liveness.step1_async_ppe.is_detecting else "Ready"
                    status_color = (255, 165, 0) if liveness.step1_async_ppe.is_detecting else (0, 255, 0)
                    cv2.putText(display_frame, f"PPE Check: {detection_status}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                    cv2.putText(display_frame, f"Next check: {30 - (liveness.step1_frame_count % 30)} frames", (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Check face positioning for Step 1
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = liveness.detector(gray)
                    
                    face_detected = False
                    face_quality_good = False
                    symmetry = 0
                    is_frontal = False
                    pose_stability = 0
                    
                    if len(faces) > 0:
                        face_detected = True
                        
                        # Check face quality for positioning
                        face = faces[0]
                        try:
                            landmarks = liveness.predictor(gray, face)
                            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
                            
                            # VERY lenient quality check for Step 1 positioning (just basic face detection)
                            symmetry, is_frontal, pose_stability = liveness.analyze_face_pose_quality(landmarks_array, frame)
                            
                            # Simple quality check from backup
                            face_quality_good = is_frontal and symmetry > 0.6 and pose_stability > 0.5
                            
                            # Draw face rectangle
                            cv2.rectangle(display_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                            
                            if face_quality_good:
                                cv2.putText(display_frame, "✅ Good face positioning", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Count frames with good positioning - SIMPLE LOGIC FROM BACKUP
                                if not hasattr(liveness, 'positioning_frames'):
                                    liveness.positioning_frames = 0
                                liveness.positioning_frames += 1
                                
                                cv2.putText(display_frame, f"Stability: {liveness.positioning_frames}/30", (10, 110), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                                
                                # Require 30 frames (1 second) of good positioning AND no PPE
                                if liveness.positioning_frames >= 30 and no_ppe_ok:
                                    print("✅ Face positioned correctly and no helmet verified!")
                                    print("🎯 Starting liveness challenge...")
                                    
                                    # Generate blink challenge now
                                    blink_challenge = liveness.generate_blink_challenge()
                                    print(f"🎯 Step 2/8: {blink_challenge['text']}")
                                    print("📸 Recording and analyzing all frames during challenge...")
                                    
                                    current_step = 1
                                    frame_count = 0
                                elif liveness.positioning_frames >= 30 and not no_ppe_ok:
                                    cv2.putText(display_frame, "⚠️ Face positioned but PPE detected", (10, 140), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 2)
                            else:
                                cv2.putText(display_frame, "⚠️ Please face camera directly", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                cv2.putText(display_frame, f"Frontal: {is_frontal}, Symmetry: {symmetry:.2f}", (10, 110), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 2)
                                # Reset counter if quality drops - SIMPLE LOGIC FROM BACKUP
                                liveness.positioning_frames = 0
                        except Exception as e:
                            face_quality_good = False
                            cv2.putText(display_frame, "⚠️ Face detection error", (10, 110), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "❌ No face detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(display_frame, "Move closer to camera", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 2)
                        # Reset counter if no face - SIMPLE LOGIC FROM BACKUP
                        liveness.positioning_frames = 0
                else:
                    current_step = 3  # Skip to recognition if no liveness
            
            # STEP 1: Liveness Challenge with Frame Recording
            elif current_step == 1:
                if use_liveness and blink_challenge:
                    required_blinks = blink_challenge['blinks']
                    
                    cv2.putText(display_frame, f"STEP 2: {blink_challenge['text']}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Recording ALL frames during challenge", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    
                    # Process blink detection with frame storage
                    blink_detected, display_frame, ear, eyes_open, blink_phase = liveness.detect_blink_with_storage(display_frame)
                    
                    # Show current phase
                    phase_color = {
                        "open": (0, 255, 0),
                        "closing": (0, 255, 255),
                        "closed": (0, 0, 255),
                        "opening": (255, 255, 0),
                        "unknown": (128, 128, 128)
                    }
                    cv2.putText(display_frame, f"Phase: {blink_phase.upper()}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color.get(blink_phase, (255, 255, 255)), 2)
                    
                    # Update progress
                    remaining = max(0, required_blinks - liveness.total_blinks)
                    if remaining > 0:
                        cv2.putText(display_frame, f"Blink {remaining} more time(s)", (10, 170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "Challenge complete! Validating timing...", (10, 170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Check if challenge completed
                    if liveness.total_blinks >= required_blinks:
                        timing_valid = liveness.validate_human_timing()
                        
                        if timing_valid:
                            print(f"✅ STEP 2 COMPLETE: Liveness verified!")
                            print(f"   Blinks: {liveness.total_blinks}, Timing: HUMAN")
                            print(f"   Frames recorded: {len(liveness.liveness_frames)}")
                            liveness_completed = True
                            current_step = 2
                        else:
                            print(f"❌ STEP 2 FAILED: Bot-like timing detected")
                            break
                else:
                    current_step = 2
            
            # STEP 2: Frame Selection from Liveness Period
            elif current_step == 2:
                cv2.putText(display_frame, "STEP 3: Selecting best frame from liveness period...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "Analyzing recorded frames for quality...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                
                if liveness_completed and use_liveness:
                    # Select best frame from liveness period
                    best_frame, selection_report = liveness.select_best_frame_from_liveness_period()
                    
                    if best_frame is not None:
                        final_frame = best_frame
                        print(f"✅ STEP 3 COMPLETE: Best frame selected from liveness period")
                        print(f"   Analysis: {selection_report['total_frames_analyzed']} frames")
                        print(f"   Quality frames: {selection_report['quality_frames_found']}")
                        print(f"   Selected quality: {selection_report['selected_frame_quality']:.1f}")
                        current_step = 3
                    else:
                        print(f"❌ STEP 3 FAILED: No suitable frame found")
                        break
                else:
                    # No liveness - use current frame
                    final_frame = frame.copy()
                    current_step = 3
            
            # STEP 3: Face Recognition with Selected Frame
            elif current_step == 3:
                cv2.putText(display_frame, "STEP 4: Face recognition with preprocessing...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "Verifying identity with enhanced preprocessing...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                
                if final_frame is not None:
                    print("🎯 Performing face recognition with preprocessing pipeline...")
                    
                    # Generate unique timestamp for this verification
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Try to get face landmarks for better preprocessing
                    gray = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
                    faces = liveness.detector(gray)
                    
                    landmarks_np = None
                    bbox = None
                    if len(faces) > 0 and liveness.predictor is not None:
                        face = faces[0]
                        landmarks = liveness.predictor(gray, face).parts()
                        landmarks_np = np.array([(p.x, p.y) for p in landmarks])
                        bbox = (face.left(), face.top(), face.left() + face.width(), face.top() + face.height())
                    
                    # Method 1: Preprocessed verification with debug stages (vs all references)
                    rgb_face = preprocess_face_crop(final_frame, bbox=bbox, landmarks=landmarks_np, 
                                                  save_stages=True, stage_prefix=f"{timestamp}_final_")
                    
                    result_preprocessed = None
                    if rgb_face is not None:
                        result_preprocessed = verify_face_against_references(rgb_face, reference_encodings, threshold=threshold)
                        print(f"🔍 Preprocessed verification: {result_preprocessed.get('match_percent', 0)}% matched -> {result_preprocessed.get('matched_name')}")
                    
                    # Method 2: Original direct verification for comparison (full-frame vs all refs)
                    rgb_full = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    result_direct = verify_face_against_references(rgb_full, reference_encodings, threshold=threshold)
                    print(f"🔍 Direct verification: {result_direct.get('match_percent', 0)}% matched -> {result_direct.get('matched_name')}")
                    
                    # Choose best by match_percent (prefer higher)
                    valid_pre = result_preprocessed if (result_preprocessed and result_preprocessed.get('success')) else None
                    valid_dir = result_direct if (result_direct and result_direct.get('success')) else None
                    if valid_pre and valid_dir:
                        if valid_pre['match_percent'] >= valid_dir['match_percent']:
                            final_result = valid_pre; method_used = "preprocessed"
                        else:
                            final_result = valid_dir; method_used = "direct"
                    else:
                        final_result = valid_pre or valid_dir or {"success": False, "error": "No valid result"}
                        method_used = "preprocessed" if valid_pre else ("direct" if valid_dir else "none")
                    
                    matched_name = final_result.get("matched_name") if final_result and final_result.get("success") else None
                    matched_reference_name = matched_name  # Store for later use
                    
                    # Save comparison frame
                    comparison_frame = final_frame.copy()
                    prep_percent = result_preprocessed.get('match_percent', 0) if result_preprocessed else 0
                    direct_percent = result_direct.get('match_percent', 0) if result_direct else 0
                    prep_name = result_preprocessed.get('matched_name', 'None') if result_preprocessed else 'None'
                    direct_name = result_direct.get('matched_name', 'None') if result_direct else 'None'
                    
                    cv2.putText(comparison_frame, f"Preprocessed: {prep_percent}% -> {prep_name}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(comparison_frame, f"Direct: {direct_percent}% -> {direct_name}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    # FRAME SAVING DISABLED - Verification comparison frame
                    # cv2.imwrite(f"Pre-Process/{timestamp}_verification_comparison.jpg", comparison_frame)
                    
                    print(f"🎯 Using {method_used} method for final result")
                    
                    if final_result.get("success"):
                        final_match_percent = final_result["match_percent"]
                        match_status = "MATCH" if final_result["match"] else "NO MATCH"
                        face_recognition_success = final_result["match"]
                        
                        # Log matched reference when a match is found
                        if face_recognition_success and matched_name:
                            fr_log = f"""
=== FACE RECOGNITION MATCH ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Matched Reference: {matched_name}
Match Percent: {final_match_percent}%
Reference Source: references/{matched_name}
Method Used: {method_used.upper()}
================================
"""
                            try:
                                with open('verification_log.txt', 'a', encoding='utf-8') as f:
                                    f.write(fr_log + '\n')
                                print(f"📝 Match logged: {matched_name} ({final_match_percent}%)")
                            except Exception as e:
                                print(f"⚠️ Failed to write face match to log: {e}")
                        
                        print(f"✅ STEP 4 COMPLETE: Face recognition complete!")
                        print(f"   Method: {method_used.upper()}")
                        print(f"   Result: {match_status} ({final_match_percent}%)")
                        print(f"   Matched: {matched_name if matched_name else 'None'}")
                        print(f"   Debug images saved to Pre-Process folder")
                        
                        # Show result on frame
                        result_color = (0, 255, 0) if final_result["match"] else (0, 0, 255)
                        cv2.putText(display_frame, f"Result: {match_status} ({final_match_percent}%)", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
                        cv2.putText(display_frame, f"Method: {method_used.upper()}", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        if matched_name:
                            cv2.putText(display_frame, f"Matched: {matched_name}", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Check if we should proceed to PPE detection
                        if face_recognition_success:
                            cv2.putText(display_frame, "✅ Face verified! Proceeding to PPE check...", (10, 180), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            print("🎯 Face recognition successful! Proceeding to PPE helmet detection...")
                            
                            # Wait a moment to show result, then proceed to PPE
                            cv2.imshow(window_name, display_frame)
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                            cv2.waitKey(2000)  # Show result for 2 seconds
                            current_step = 4  # Proceed to PPE detection
                        else:
                            cv2.putText(display_frame, "❌ Face verification failed - stopping here", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            print("❌ Face recognition failed - PPE detection will be skipped")
                            
                            # Wait a moment to show result
                            cv2.imshow(window_name, display_frame)
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                            cv2.waitKey(2000)  # Show result for 2 seconds
                            break
                    else:
                        print(f"❌ STEP 4 FAILED: Face recognition error - {final_result.get('error', 'Unknown')}")
                        break
                else:
                    print(f"❌ STEP 4 FAILED: No frame selected for recognition")
                    break
            
            # STEP 4: PPE Helmet Detection (only if face recognition successful)
            elif current_step == 4:
                if face_recognition_success:
                    cv2.putText(display_frame, "STEP 5: PPE Helmet Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Please wear your helmet now", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # Initialize step 5 variables
                    if not hasattr(liveness, 'step5_initialized'):
                        liveness.step5_initialized = True
                        liveness.step5_frame_count = 0
                        liveness.async_ppe_detector = AsyncPPEDetector()
                        # FRAME SAVING DISABLED - Reset for new session
                        # liveness.async_ppe_detector.reset_saved_items()  # Reset for new session
                        liveness.last_helmet_result = {"helmet_detected": False, "person_detected": False}
                        print("🎯 Step 5/8: PPE Helmet Detection started")
                        print("⚠️ Please wear your helmet for safety verification")
                    
                    liveness.step5_frame_count += 1
                    
                    # Start async PPE detection every 30 frames (non-blocking)
                    if liveness.step5_frame_count % 30 == 0:
                        print(f"🔍 Starting async PPE detection check #{liveness.step5_frame_count // 30}...")
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.async_ppe_detector.start_detection(frame)  # capture_frame=True, step_name="Helmet_Detection"
                    
                    # Check for async detection results (non-blocking)
                    async_result = liveness.async_ppe_detector.get_result()
                    if async_result is not None:
                        if async_result.get("success", False):
                            helmet_detected = async_result.get("helmet_detected", False)
                            person_detected = async_result.get("person_detected", False)
                            
                            # Update last result
                            liveness.last_helmet_result = {
                                "helmet_detected": helmet_detected,
                                "person_detected": person_detected
                            }
                            
                            if helmet_detected:
                                print("✅ Helmet detected! Moving to gloves detection...")
                                
                                # Log helmet success
                                log_entry = f"""
=== PPE HELMET DETECTION SUCCESS ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Result: HELMET DETECTED ✅
Helmet Detected: {helmet_detected}
Person Detected: {person_detected}
================================"""
                                
                                with open('verification_log.txt', 'a', encoding='utf-8') as f:
                                    f.write(log_entry + '\n')
                                
                                ppe_completed = True
                                
                                # FRAME SAVING DISABLED - Log capture summary
                                # capture_summary = liveness.async_ppe_detector.get_capture_summary()
                                # print(f"📸 Helmet Detection Frames Summary:")
                                # print(f"   Step frame saved: {capture_summary['step_frame_saved']}")
                                # print(f"   Success frames: {capture_summary['success_items_saved']}")
                                
                                current_step = 5  # Move to safety glasses detection
                                
                                # Show success message
                                cv2.putText(display_frame, "✅ HELMET DETECTED!", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.putText(display_frame, "Moving to gloves check...", (10, 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                print(f"❌ No helmet detected - H:{helmet_detected} P:{person_detected}")
                        else:
                            print(f"⚠️ PPE detection error: {async_result.get('error', 'Unknown')}")
                    
                    # Show current status based on last result
                    helmet_status = liveness.last_helmet_result.get("helmet_detected", False)
                    person_status = liveness.last_helmet_result.get("person_detected", False)
                    
                    if helmet_status:
                        cv2.putText(display_frame, "✅ Helmet detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "❌ No helmet detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(display_frame, "Please wear your helmet", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Show frame count and detection status
                    cv2.putText(display_frame, f"Frame: {liveness.step5_frame_count}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Next check: {30 - (liveness.step5_frame_count % 30)} frames", (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Show async detection status
                    detection_status = "Processing..." if liveness.async_ppe_detector.is_detecting else "Ready"
                    status_color = (255, 165, 0) if liveness.async_ppe_detector.is_detecting else (0, 255, 0)
                    cv2.putText(display_frame, f"Detection: {detection_status}", (10, 490), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                else:
                    print("❌ Cannot start PPE detection - face recognition failed")
                    print("🔄 Returning to Step 1 - No-Helmet Check + Face Positioning...")
                    # Reset all state variables for new attempt
                    current_step = 0
                    final_frame = None
                    final_match_percent = 0
                    liveness_completed = False
                    face_recognition_success = False
                    ppe_completed = False
                    gloves_completed = False
                    vest_completed = False
                    boots_completed = False
                    glasses_completed = False
                    mask_completed = False
                    matched_reference_name = None
                    # Clear any existing detector states
                    if hasattr(liveness, 'step5_initialized'):
                        delattr(liveness, 'step5_initialized')
                    if hasattr(liveness, 'step5_glasses_initialized'):
                        delattr(liveness, 'step5_glasses_initialized')
                    if hasattr(liveness, 'step6_mask_initialized'):
                        delattr(liveness, 'step6_mask_initialized')
                    if hasattr(liveness, 'step7_gloves_initialized'):
                        delattr(liveness, 'step7_gloves_initialized')
                    if hasattr(liveness, 'step8_ocr_initialized'):
                        delattr(liveness, 'step8_ocr_initialized')
                    if hasattr(liveness, 'step7_initialized'):
                        delattr(liveness, 'step7_initialized')
                    if hasattr(liveness, 'step8_initialized'):
                        delattr(liveness, 'step8_initialized')
                    continue
            
            # STEP 5: Safety Glasses Detection (only if helmet detection successful)
            elif current_step == 5:
                if ppe_completed:
                    cv2.putText(display_frame, "STEP 5: Safety Glasses Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Wear your safety glasses", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # Initialize step 5 variables
                    if not hasattr(liveness, 'step5_glasses_initialized'):
                        liveness.step5_glasses_initialized = True
                        liveness.step5_glasses_frame_count = 0
                        liveness.async_glasses_detector = AsyncPPEDetector()
                        # FRAME SAVING DISABLED - Reset for new session
                        # liveness.async_glasses_detector.reset_saved_items()  # Reset for new session
                        liveness.last_glasses_result = {"glasses_detected": False, "person_detected": False}
                        print("🎯 Step 5/10: Safety Glasses Detection started")
                        print("👓 Please wear your safety glasses")
                    
                    liveness.step5_glasses_frame_count += 1
                    
                    # Start async glasses detection every 30 frames
                    if liveness.step5_glasses_frame_count % 30 == 0:
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.async_glasses_detector.start_detection(frame)  # capture_frame=True, step_name="Glasses_Detection"
                    
                    # Check for async detection results
                    async_result = liveness.async_glasses_detector.get_result()
                    if async_result is not None:
                        if async_result.get("success", False):
                            glasses_detected = async_result.get("glasses_detected", False)
                            
                            liveness.last_glasses_result = {
                                "glasses_detected": glasses_detected,
                                "person_detected": async_result.get("person_detected", False),
                                "ppe_predictions": async_result.get("ppe_predictions", {"predictions": []})
                            }
                            
                            if glasses_detected:
                                print("✅ Safety glasses detected! Moving to mask detection...")
                                glasses_completed = True
                                
                                # FRAME SAVING DISABLED - Log capture summary
                                # capture_summary = liveness.async_glasses_detector.get_capture_summary()
                                # print(f"📸 Glasses Detection Frames Summary:")
                                # print(f"   Step frame saved: {capture_summary['step_frame_saved']}")
                                # print(f"   Success frames: {capture_summary['success_items_saved']}")
                                
                                current_step = 6  # Move to mask detection
                            else:
                                print(f"❌ No safety glasses detected")
                        else:
                            print(f"⚠️ Glasses detection error: {async_result.get('error', 'Unknown')}")
                    
                    # Draw PPE bounding boxes if available
                    if hasattr(liveness, 'last_glasses_result') and 'ppe_predictions' in liveness.last_glasses_result:
                        display_frame = draw_predictions(display_frame, liveness.last_glasses_result['ppe_predictions'])
                    
                    # Show current status
                    glasses_status = liveness.last_glasses_result.get("glasses_detected", False)
                    
                    if glasses_status:
                        cv2.putText(display_frame, "✅ Safety glasses detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "❌ No safety glasses detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(display_frame, "👓 Please wear safety glasses", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.putText(display_frame, f"Frame: {liveness.step5_glasses_frame_count}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Next check: {30 - (liveness.step5_glasses_frame_count % 30)} frames", (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    print("❌ Cannot start glasses detection - helmet detection not completed")
                    print("🔄 Returning to Step 1...")
                    current_step = 0
                    continue
            
            # STEP 6: Safety Mask Detection (only if glasses detection successful)
            elif current_step == 6:
                if glasses_completed:
                    cv2.putText(display_frame, "STEP 6: Safety Mask Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Wear your safety mask", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # Initialize step 6 variables
                    if not hasattr(liveness, 'step6_mask_initialized'):
                        liveness.step6_mask_initialized = True
                        liveness.step6_mask_frame_count = 0
                        liveness.async_mask_detector = AsyncPPEDetector()
                        # FRAME SAVING DISABLED - Reset for new session
                        # liveness.async_mask_detector.reset_saved_items()  # Reset for new session
                        liveness.last_mask_result = {"mask_detected": False, "person_detected": False}
                        print("🎯 Step 6/10: Safety Mask Detection started")
                        print("😷 Please wear your safety mask")
                    
                    liveness.step6_mask_frame_count += 1
                    
                    # Start async mask detection every 30 frames
                    if liveness.step6_mask_frame_count % 30 == 0:
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.async_mask_detector.start_detection(frame)  # capture_frame=True, step_name="Mask_Detection"
                    
                    # Check for async detection results
                    async_result = liveness.async_mask_detector.get_result()
                    if async_result is not None:
                        if async_result.get("success", False):
                            mask_detected = async_result.get("mask_detected", False)
                            
                            liveness.last_mask_result = {
                                "mask_detected": mask_detected,
                                "person_detected": async_result.get("person_detected", False),
                                "ppe_predictions": async_result.get("ppe_predictions", {"predictions": []})
                            }
                            
                            if mask_detected:
                                print("✅ Safety mask detected! Moving to gloves detection...")
                                mask_completed = True
                                
                                # FRAME SAVING DISABLED - Log capture summary
                                # capture_summary = liveness.async_mask_detector.get_capture_summary()
                                # print(f"📸 Mask Detection Frames Summary:")
                                # print(f"   Step frame saved: {capture_summary['step_frame_saved']}")
                                # print(f"   Success frames: {capture_summary['success_items_saved']}")
                                
                                current_step = 7  # Move to gloves detection
                            else:
                                print(f"❌ No safety mask detected")
                        else:
                            print(f"⚠️ Mask detection error: {async_result.get('error', 'Unknown')}")
                    
                    # Draw PPE bounding boxes if available
                    if hasattr(liveness, 'last_mask_result') and 'ppe_predictions' in liveness.last_mask_result:
                        display_frame = draw_predictions(display_frame, liveness.last_mask_result['ppe_predictions'])
                    
                    # Show current status
                    mask_status = liveness.last_mask_result.get("mask_detected", False)
                    
                    if mask_status:
                        cv2.putText(display_frame, "✅ Safety mask detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "❌ No safety mask detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(display_frame, "😷 Please wear safety mask", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.putText(display_frame, f"Frame: {liveness.step6_mask_frame_count}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Next check: {30 - (liveness.step6_mask_frame_count % 30)} frames", (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    print("❌ Cannot start mask detection - glasses detection not completed")
                    print("🔄 Returning to Step 1...")
                    current_step = 0
                    continue
            
            # STEP 7: Safety Gloves Detection (only if mask detection successful)
            elif current_step == 7:
                if mask_completed:
                    cv2.putText(display_frame, "STEP 7: Safety Gloves Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Raise your hands to show gloves", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # Initialize step 7 variables
                    if not hasattr(liveness, 'step7_gloves_initialized'):
                        liveness.step7_gloves_initialized = True
                        liveness.step7_gloves_frame_count = 0
                        liveness.async_gloves_detector = AsyncPPEDetector()
                        # FRAME SAVING DISABLED - Reset for new session
                        # liveness.async_gloves_detector.reset_saved_items()  # Reset for new session
                        liveness.last_gloves_result = {"gloves_detected": False, "person_detected": False}
                        print("🎯 Step 7/10: Safety Gloves Detection started")
                        print("✋ Please raise your hands to show your safety gloves")
                    
                    liveness.step7_gloves_frame_count += 1
                    
                    # Start async gloves detection every 30 frames
                    if liveness.step7_gloves_frame_count % 30 == 0:
                        print(f"🔍 Starting async gloves detection check #{liveness.step7_gloves_frame_count // 30}...")
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.async_gloves_detector.start_detection(frame)  # capture_frame=True, step_name="Gloves_Detection"
                    
                    # Check for async detection results
                    async_result = liveness.async_gloves_detector.get_result()
                    if async_result is not None:
                        if async_result.get("success", False):
                            # ✅ Use async results directly (no extra YOLO call)
                            gloves_detected = async_result.get("gloves_detected", False)
                            person_detected = async_result.get("person_detected", False)
                            
                            liveness.last_gloves_result = {
                                "gloves_detected": gloves_detected,
                                "person_detected": person_detected,
                                "ppe_predictions": async_result.get("ppe_predictions", {"predictions": []})
                            }
                            
                            if gloves_detected:
                                print("✅ Safety gloves detected! Moving to OCR detection...")
                                
                                # Log gloves success
                                log_entry = f"""
=== PPE GLOVES DETECTION SUCCESS ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Result: SAFETY GLOVES DETECTED ✅
Gloves Detected: {gloves_detected}
Person Detected: {person_detected}
================================"""
                                
                                with open('verification_log.txt', 'a', encoding='utf-8') as f:
                                    f.write(log_entry + '\n')
                                
                                gloves_completed = True
                                current_step = 8  # Move to OCR step
                            else:
                                print(f"❌ No safety gloves detected - G:{gloves_detected} P:{person_detected}")
                        else:
                            print(f"⚠️ Gloves detection error: {async_result.get('error', 'Unknown')}")
                    
                    # Draw PPE bounding boxes if available
                    if hasattr(liveness, 'last_gloves_result') and 'ppe_predictions' in liveness.last_gloves_result:
                        display_frame = draw_predictions(display_frame, liveness.last_gloves_result['ppe_predictions'])
                    
                    # Show current status
                    gloves_status = liveness.last_gloves_result.get("gloves_detected", False)
                    
                    if gloves_status:
                        cv2.putText(display_frame, "✅ Safety gloves detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "❌ No safety gloves detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(display_frame, "✋ Raise hands to show gloves", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Show frame count and detection status
                    cv2.putText(display_frame, f"Frame: {liveness.step7_gloves_frame_count}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Next check: {30 - (liveness.step7_gloves_frame_count % 30)} frames", (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                        print("❌ Cannot start gloves detection - helmet detection not completed")
                        print("🔄 Returning to Step 1...")
                        current_step = 0
                        continue
            
            # STEP 8: OCR Vest Number Detection (only if gloves detection successful) 
            elif current_step == 8:
                if gloves_completed:
                    cv2.putText(display_frame, "STEP 8: Vest Number OCR", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Show your vest number clearly", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # Initialize OCR step variables
                    if not hasattr(liveness, 'step8_ocr_initialized'):
                        liveness.step8_ocr_initialized = True
                        liveness.step8_ocr_frame_count = 0
                        liveness.async_ocr_detector = AsyncOCRDetector()
                        # FRAME SAVING DISABLED - Reset for new session
                        # liveness.async_ocr_detector.reset_saved_items()
                        liveness.last_ocr_result = {"vest_number": None, "success": False}
                        liveness.vest_number_detected = None
                        print("🎯 Step 8/10: OCR Vest Number Detection started")
                        print("📋 Please show your vest number clearly to the camera")
                    
                    liveness.step8_ocr_frame_count += 1
                    
                    # Start async OCR detection every 30 frames
                    if liveness.step8_ocr_frame_count % 30 == 0:
                        print(f"🔍 Starting async OCR detection #{liveness.step8_ocr_frame_count // 30}...")
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.async_ocr_detector.start_detection(frame)  # capture_frame=True, step_name="Vest_Number_OCR"
                    
                    # Check for async OCR results
                    async_result = liveness.async_ocr_detector.get_result()
                    if async_result is not None:
                        if async_result.get("success", False):
                            vest_number = async_result.get("vest_number")
                            confidence = async_result.get("best_confidence", 0)
                            
                            liveness.last_ocr_result = {
                                "vest_number": vest_number,
                                "confidence": confidence,
                                "success": True,
                                "total_detections": async_result.get("total_detections", 0)
                            }
                            
                            if vest_number and confidence > 0.6:  # Require good confidence
                                if liveness.vest_number_detected != vest_number:
                                    liveness.vest_number_detected = vest_number
                                    print(f"📋 Vest number detected: {vest_number} (confidence: {confidence:.3f})")
                        else:
                            print(f"⚠️ OCR detection error: {async_result.get('error', 'Unknown')}")
                    
                    # Show OCR status
                    vest_number = liveness.last_ocr_result.get("vest_number")
                    ocr_confidence = liveness.last_ocr_result.get("confidence", 0)
                    
                    if vest_number and ocr_confidence > 0.6:
                        cv2.putText(display_frame, f"✅ Number: {vest_number}", (10, 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Confidence: {ocr_confidence*100:.1f}%", (10, 170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Auto-proceed after 3 seconds of stable detection
                        if not hasattr(liveness, 'ocr_stable_start'):
                            liveness.ocr_stable_start = time.time()
                        elif time.time() - liveness.ocr_stable_start >= 3.0:
                            print(f"✅ OCR detection completed! Vest number: {vest_number}")
                            current_step = 9  # Move to vest detection
                    else:
                        cv2.putText(display_frame, "❌ No clear number detected", (10, 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(display_frame, "Position vest number clearly", (10, 170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                        # Reset stable timer
                        if hasattr(liveness, 'ocr_stable_start'):
                            delattr(liveness, 'ocr_stable_start')
                    
                    # Show async detection status
                    detection_status = "Processing..." if liveness.async_ocr_detector.is_detecting else "Ready"
                    status_color = (255, 165, 0) if liveness.async_ocr_detector.is_detecting else (0, 255, 0)
                    cv2.putText(display_frame, f"OCR: {detection_status}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                    
                else:
                    print("❌ Cannot start OCR detection - gloves detection not completed")
                    print("🔄 Returning to Step 1...")
                    current_step = 0
                    continue
            
            # STEP 9: Safety Vest Detection (only if OCR completed)
            elif current_step == 9:
                if gloves_completed:
                    cv2.putText(display_frame, "STEP 9: Safety Vest Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Step back to show your vest", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # Initialize step 7 variables
                    if not hasattr(liveness, 'step7_initialized'):
                        liveness.step7_initialized = True
                        liveness.step7_frame_count = 0
                        liveness.async_vest_detector = AsyncPPEDetector()
                        # FRAME SAVING DISABLED - Reset for new session
                        # liveness.async_vest_detector.reset_saved_items()  # Reset for new session
                        liveness.last_vest_result = {"vest_detected": False, "person_detected": False}
                        print("🎯 Step 7/8: Safety Vest Detection started")
                        print("👕 Please step back to show your safety vest")
                    
                    liveness.step7_frame_count += 1
                    
                    # Start async vest detection every 30 frames
                    if liveness.step7_frame_count % 30 == 0:
                        print(f"🔍 Starting async vest detection check #{liveness.step7_frame_count // 30}...")
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.async_vest_detector.start_detection(frame)  # capture_frame=True, step_name="Vest_Detection"
                    
                    # Check for async detection results
                    async_result = liveness.async_vest_detector.get_result()
                    if async_result is not None:
                        if async_result.get("success", False):
                            # ✅ Use async results directly (no extra YOLO call)
                            vest_detected = async_result.get("vest_detected", False)
                            person_detected = async_result.get("person_detected", False)
                            
                            liveness.last_vest_result = {
                                "vest_detected": vest_detected,
                                "person_detected": person_detected,
                                "ppe_predictions": async_result.get("ppe_predictions", {"predictions": []})
                            }
                            
                            if vest_detected:
                                print("✅ Safety vest detected! Moving to boots detection...")
                                
                                # Log vest success
                                log_entry = f"""
=== PPE VEST DETECTION SUCCESS ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Result: SAFETY VEST DETECTED ✅
Vest Detected: {vest_detected}
Person Detected: {person_detected}
================================"""
                                
                                with open('verification_log.txt', 'a', encoding='utf-8') as f:
                                    f.write(log_entry + '\n')
                                
                                vest_completed = True
                                current_step = 10  # Move to boots detection
                            else:
                                print(f"❌ No safety vest detected - V:{vest_detected} P:{person_detected}")
                        else:
                            print(f"⚠️ Vest detection error: {async_result.get('error', 'Unknown')}")
                    
                    # Draw PPE bounding boxes if available
                    if hasattr(liveness, 'last_vest_result') and 'ppe_predictions' in liveness.last_vest_result:
                        display_frame = draw_predictions(display_frame, liveness.last_vest_result['ppe_predictions'])
                    
                    # Show current status
                    vest_status = liveness.last_vest_result.get("vest_detected", False)
                    
                    if vest_status:
                        cv2.putText(display_frame, "✅ Safety vest detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "❌ No safety vest detected", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(display_frame, "👕 Step back to show vest", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Show frame count and detection status
                    cv2.putText(display_frame, f"Frame: {liveness.step7_frame_count}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Next check: {30 - (liveness.step7_frame_count % 30)} frames", (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    print("❌ Cannot start vest detection - gloves detection not completed")
                    print("🔄 Returning to Step 1...")
                    current_step = 0
                    continue
            
            # STEP 10: Safety Boots Detection (final PPE step)
            elif current_step == 10:
                if vest_completed:
                    cv2.putText(display_frame, "STEP 10: Safety Boots Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Show your safety boots clearly", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # Initialize step 8 variables
                    if not hasattr(liveness, 'step8_initialized'):
                        liveness.step8_initialized = True
                        liveness.step8_frame_count = 0
                        liveness.async_boots_detector = AsyncPPEDetector()
                        # FRAME SAVING DISABLED - Reset for new session
                        # liveness.async_boots_detector.reset_saved_items()  # Reset for new session
                        liveness.last_boots_result = {"boots_detected": False, "person_detected": False}
                        print("🎯 Step 8/8: Safety Boots Detection started")
                        print("🥾 Please show your safety boots clearly")
                    
                    liveness.step8_frame_count += 1
                    
                    # Start async PPE detection every 30 frames for boots
                    if liveness.step8_frame_count % 30 == 0:
                        # FRAME SAVING DISABLED - Start detection without capture
                        liveness.async_boots_detector.start_detection(frame)  # capture_frame=True, step_name="Boots_Detection"
                    
                    # Check for async detection results (non-blocking)
                    async_result = liveness.async_boots_detector.get_result()
                    if async_result is not None:
                        if async_result.get("success", False):
                            boots_detected = async_result.get("boots_detected", False)
                            person_detected = async_result.get("person_detected", False)
                            liveness.last_boots_result = {
                                "boots_detected": boots_detected,
                                "person_detected": person_detected,
                                "ppe_predictions": async_result.get("ppe_predictions", {"predictions": []})
                            }
                            if boots_detected:  # Only require boots detection for success
                                print("✅ Safety boots detected!")
                                boots_completed = True
                                # Show final success message
                                cv2.putText(display_frame, "✅ SAFETY BOOTS DETECTED!", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.putText(display_frame, "All PPE verifications complete!", (10, 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(display_frame, "🔄 Restarting in 3 seconds...", (10, 150), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                cv2.imshow(window_name, display_frame)
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                                cv2.waitKey(3000)
                                
                                # Log the successful completion
                                print("✅ All PPE verifications complete! Logging results...")
                                
                                # Determine final results for logging
                                liveness_verified = liveness_completed and timing_valid
                                face_verified = final_match_percent > 0 and final_match_percent >= threshold
                                anti_cheat_passed = liveness_verified
                                all_ppe_complete = ppe_completed and glasses_completed and mask_completed and gloves_completed and vest_completed and boots_completed
                                
                                # Log the successful verification
                                vest_number = getattr(liveness, 'vest_number_detected', None)
                                log_verification_result(face_verified, final_match_percent, liveness_verified, 
                                                      reference_path, threshold, anti_cheat_passed, "SUCCESS", 
                                                      ppe_completed, glasses_completed, mask_completed, gloves_completed, vest_completed, boots_completed, matched_reference_name, vest_number)
                                
                                print("🔄 Returning to Step 1 for next verification...")
                                
                                # Reset all state variables for new attempt
                                current_step = 0
                                final_frame = None
                                final_match_percent = 0
                                liveness_completed = False
                                face_recognition_success = False
                                ppe_completed = False
                                gloves_completed = False
                                vest_completed = False
                                boots_completed = False
                                glasses_completed = False
                                mask_completed = False
                                matched_reference_name = None
                                
                                # Clear any existing detector states
                                if hasattr(liveness, 'step5_initialized'):
                                    delattr(liveness, 'step5_initialized')
                                if hasattr(liveness, 'step5_glasses_initialized'):
                                    delattr(liveness, 'step5_glasses_initialized')
                                if hasattr(liveness, 'step6_mask_initialized'):
                                    delattr(liveness, 'step6_mask_initialized')
                                if hasattr(liveness, 'step7_gloves_initialized'):
                                    delattr(liveness, 'step7_gloves_initialized')
                                if hasattr(liveness, 'step8_ocr_initialized'):
                                    delattr(liveness, 'step8_ocr_initialized')
                                if hasattr(liveness, 'step7_initialized'):
                                    delattr(liveness, 'step7_initialized')
                                if hasattr(liveness, 'step8_initialized'):
                                    delattr(liveness, 'step8_initialized')
                                
                                # Reset liveness detector state
                                liveness.liveness_frames = []
                                liveness.total_blinks = 0
                                liveness.blink_timestamps = []
                                if hasattr(liveness, 'positioning_frames'):
                                    liveness.positioning_frames = 0
                                
                                continue
                            else:
                                print(f"❌ No safety boots detected - B:{boots_detected}")
                        else:
                            print(f"⚠️ Boots detection error: {async_result.get('error', 'Unknown')}")
                else:
                    print("❌ Cannot start boots detection - vest detection not completed")
                    print("🔄 Returning to Step 1...")
                    current_step = 0
                    continue
            
            # Show live feed
            cv2.imshow(window_name, display_frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            
            # Allow exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("❌ Process cancelled by user")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Determine final results
        liveness_verified = liveness_completed and timing_valid
        face_verified = final_match_percent > 0 and final_match_percent >= threshold
        anti_cheat_passed = liveness_verified
        
        # Display final results
        print("\n" + "="*70)
        print("🎯 ENHANCED VERIFICATION WITH PPE DETECTION COMPLETE!")
        print("="*70)
        print(f"📋 Step 1 (No-Helmet + Face Positioning): {'✅ PASSED' if no_helmet_verified or not use_liveness else '❌ FAILED'}")
        print(f"📋 Step 2 (Liveness Challenge): {'✅ PASSED' if liveness_verified else '❌ FAILED'}")
        print(f"📋 Step 3 (Frame Selection): {'✅ PASSED' if final_frame is not None else '❌ FAILED'}")
        print(f"📋 Step 4 (Face Recognition): {'✅ PASSED' if face_verified else '❌ FAILED'} ({final_match_percent}%)")
        print(f"📋 Step 4 (PPE Helmet Detection): {'✅ PASSED' if ppe_completed else '❌ FAILED/SKIPPED'}")
        print(f"📋 Step 7 (Safety Gloves Detection): {'✅ PASSED' if gloves_completed else '❌ FAILED/SKIPPED'}")
        print(f"📋 Step 8 (Safety Vest Detection): {'✅ PASSED' if vest_completed else '❌ FAILED/SKIPPED'}")
        print(f"📋 Step 5 (Safety Glasses Detection): {'✅ PASSED' if glasses_completed else '❌ FAILED/SKIPPED'}")
        print(f"📋 Step 6 (Safety Mask Detection): {'✅ PASSED' if mask_completed else '❌ FAILED/SKIPPED'}")
        print(f"📋 Step 9 (Safety Boots Detection): {'✅ PASSED' if boots_completed else '❌ FAILED/SKIPPED'}")
        
        if selection_report:
            print(f"📊 Frame Selection Report:")
            print(f"   Total frames analyzed: {selection_report['total_frames_analyzed']}")
            print(f"   Quality frames found: {selection_report['quality_frames_found']}")
            print(f"   Selected frame quality: {selection_report['selected_frame_quality']:.1f}")
            print(f"   Selection method: {selection_report['selection_method']}")
        
        print(f"🛡️ Anti-cheat verification: {'✅ PASSED' if anti_cheat_passed else '❌ FAILED'}")
        print(f"⚠️ PPE Detection Status:")
        print(f"   🪖 Helmet: {'✅ VERIFIED' if ppe_completed else '❌ NOT DETECTED'}")
        print(f"   🥽 Glasses: {'✅ VERIFIED' if glasses_completed else '❌ NOT DETECTED'}")
        print(f"   😷 Mask: {'✅ VERIFIED' if mask_completed else '❌ NOT DETECTED'}")
        print(f"   🧤 Gloves: {'✅ VERIFIED' if gloves_completed else '❌ NOT DETECTED'}")
        print(f"   👕 Vest: {'✅ VERIFIED' if vest_completed else '❌ NOT DETECTED'}")
        print(f"   🥾 Boots: {'✅ VERIFIED' if boots_completed else '❌ NOT DETECTED'}")
        
        # Check if all PPE is complete
        all_ppe_complete = ppe_completed and glasses_completed and mask_completed and gloves_completed and vest_completed and boots_completed
        print(f"🛡️ Complete PPE Verification: {'✅ ALL PPE VERIFIED' if all_ppe_complete else '❌ INCOMPLETE PPE'}")
        
        # FRAME SAVING DISABLED - Create temporary file
        # temp_dir = tempfile.gettempdir()
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # final_image_path = os.path.join(temp_dir, f"enhanced_verification_{timestamp}.jpg")
        # if final_frame is not None:
        #     cv2.imwrite(final_image_path, final_frame)
        # else:
        #     final_image_path = None
        final_image_path = None  # No temporary file creation
        
        # Get detected vest number
        vest_number = getattr(liveness, 'vest_number_detected', None)
        
        return final_image_path, face_verified, liveness_verified, final_match_percent, anti_cheat_passed, ppe_completed, glasses_completed, mask_completed, gloves_completed, vest_completed, boots_completed, matched_reference_name, vest_number
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return None, False, False, 0, False, False, False, False, False, False, False, None

def log_verification_result(face_match, match_percent, liveness_verified, reference_path, threshold, anti_cheat_passed, overall_result, ppe_completed=False, glasses_completed=False, mask_completed=False, gloves_completed=False, vest_completed=False, boots_completed=False, matched_reference=None, vest_number=None):
    """Log verification results to a file"""
    log_file = "verification_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if overall_result == "SUCCESS":
        status = "✅ COMPLETE SUCCESS"
    elif overall_result == "PARTIAL":
        status = "⚠️ PARTIAL SUCCESS"
    else:
        status = "❌ FAILED"
    
    log_entry = f"""
=== ENHANCED VERIFICATION WITH PPE DETECTION LOG ===
Date: {timestamp}
Reference: {reference_path}
Matched Reference: {matched_reference if matched_reference else 'None'}
Face Match: {face_match}
Match Percentage: {match_percent}%
Threshold Used: {threshold}%
Liveness Verified: {liveness_verified}
Anti-Cheat Passed: {anti_cheat_passed}
PPE Helmet Detection: {ppe_completed}
PPE Glasses Detection: {glasses_completed}
PPE Mask Detection: {mask_completed}
PPE Gloves Detection: {gloves_completed}
PPE Vest Detection: {vest_completed}
PPE Vest Number: {vest_number if vest_number else 'Not detected'}
PPE Boots Detection: {boots_completed}
All PPE Complete: {ppe_completed and glasses_completed and mask_completed and gloves_completed and vest_completed and boots_completed}
Overall Result: {overall_result}
Status: {status}
Verification Type: Enhanced with Complete PPE Detection
Method: No-helmet verification → Liveness → Frame selection → Face recognition → Helmet → Gloves → Vest → Boots detection
{'='*70}
"""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(f"📝 Verification logged to {log_file}")
    except Exception as e:
        print(f"⚠️ Failed to write log: {e}")

def main():
    """Main function for enhanced verification with complete PPE detection"""
    print("=== ENHANCED VERIFICATION SYSTEM WITH COMPLETE PPE DETECTION ===")
    print("⚠️ Step 1: NO-HELMET verification (must remove helmet first)")
    print("📸 Step 2-3: Records ALL frames during liveness detection")
    print("🎯 Step 4: Selects BEST frame after liveness confirmation")
    print("👁️ Step 5: Uses highest quality eyes-open frontal frame for recognition")
    print("⛑️ Step 6: PPE Helmet detection (only if face recognition successful)")
    print("🧤 Step 7: Safety Gloves detection (raise hands to show gloves)")
    print("👕 Step 8: Safety Vest detection (step back to show vest)")
    print("🥾 Step 9: Safety Boots detection (show boots)")
    print("🛡️ Enterprise-grade anti-cheat with comprehensive safety verification")
    
    # Check if reference image exists
    reference_path = "reference.jpg"
    if not os.path.exists(reference_path):
        print(f"❌ Error: Reference image '{reference_path}' not found!")
        print("Please ensure reference.jpg is in the same directory as this script.")
        return
    
    # Get settings from user
    threshold = 50.0
    try:
        user_input = input(f"Enter match threshold (0-100, default {threshold}): ").strip()
        if user_input:
            threshold = float(user_input)
            if threshold < 0 or threshold > 100:
                print("Invalid threshold. Using default value of 50.")
                threshold = 50.0
    except ValueError:
        print("Invalid input. Using default threshold of 50.")
    
    print("\n🎯 Process flow:")
    print("1️⃣ Remove helmet + face positioning")
    print("2️⃣ Blink 3 times for liveness (records all frames)")
    print("3️⃣ Select best frame from liveness period")
    print("4️⃣ Face recognition with selected frame")
    print("5️⃣ Wear helmet for PPE detection (only if face verification successful)")
    print("📸 3-second intervals for PPE detection")
    
    print(f"\nStarting enhanced verification with PPE detection...")
    
    # Perform verification
    result = liveness_with_frame_selection_verification(reference_path, threshold)
    
    if result is None or len(result) < 5:
        print("No verification completed. Exiting.")
        return
    
    # Handle both old and new return formats
    if len(result) == 13:
        captured_path, face_verified, liveness_verified, match_percent, anti_cheat_passed, ppe_completed, glasses_completed, mask_completed, gloves_completed, vest_completed, boots_completed, matched_reference, vest_number = result
    elif len(result) == 12:
        captured_path, face_verified, liveness_verified, match_percent, anti_cheat_passed, ppe_completed, glasses_completed, mask_completed, gloves_completed, vest_completed, boots_completed, matched_reference = result
        vest_number = None  # Default for backward compatibility
    elif len(result) == 11:
        captured_path, face_verified, liveness_verified, match_percent, anti_cheat_passed, ppe_completed, glasses_completed, mask_completed, gloves_completed, vest_completed, boots_completed = result
        matched_reference = vest_number = None  # Default for backward compatibility
    elif len(result) == 9:
        captured_path, face_verified, liveness_verified, match_percent, anti_cheat_passed, ppe_completed, gloves_completed, vest_completed, boots_completed, matched_reference = result
        glasses_completed = mask_completed = False  # Default for backward compatibility
        vest_number = None
    elif len(result) == 8:
        captured_path, face_verified, liveness_verified, match_percent, anti_cheat_passed, ppe_completed, gloves_completed, vest_completed = result
        glasses_completed = mask_completed = boots_completed = False  # Default for backward compatibility
        matched_reference = vest_number = None
    elif len(result) == 6:
        captured_path, face_verified, liveness_verified, match_percent, anti_cheat_passed, ppe_completed = result
        glasses_completed = mask_completed = gloves_completed = vest_completed = boots_completed = False  # Default for backward compatibility
        matched_reference = vest_number = None
    else:
        captured_path, face_verified, liveness_verified, match_percent, anti_cheat_passed = result
        ppe_completed = glasses_completed = mask_completed = gloves_completed = vest_completed = boots_completed = False  # Default for backward compatibility
        matched_reference = vest_number = None
    
    if captured_path is None:
        print("No verification completed. Exiting.")
        return
    
    # Check if all PPE is complete
    all_ppe_complete = ppe_completed and glasses_completed and mask_completed and gloves_completed and vest_completed and boots_completed
    
    # Display final results
    print("\n" + "="*70)
    print("🎯 FINAL ENHANCED VERIFICATION RESULTS")
    print("="*70)
    
    if face_verified and liveness_verified and anti_cheat_passed and all_ppe_complete:
        print("🎉 COMPLETE VERIFICATION SUCCESS!")
        print("✅ No-Helmet Check: PASSED")
        print("✅ Liveness Check: PASSED")
        print("✅ Frame Selection: PASSED")
        print("✅ Face Recognition: PASSED")
        print("✅ PPE Helmet Detection: PASSED")
        print("✅ PPE Gloves Detection: PASSED")
        print("✅ PPE Vest Detection: PASSED")
        print("✅ PPE Boots Detection: PASSED")
        overall_result = "SUCCESS"
    elif face_verified and liveness_verified and anti_cheat_passed and ppe_completed:
        print("⚠️ PARTIAL SUCCESS - PPE INCOMPLETE")
        print("✅ No-Helmet Check: PASSED")
        print("✅ Liveness Check: PASSED")
        print("✅ Face Recognition: PASSED")
        print("✅ PPE Helmet Detection: PASSED")
        print(f"❌ PPE Gloves Detection: {'PASSED' if gloves_completed else 'FAILED'}")
        print(f"❌ PPE Vest Detection: {'PASSED' if vest_completed else 'FAILED'}")
        print(f"❌ PPE Boots Detection: {'PASSED' if boots_completed else 'FAILED'}")
        overall_result = "PARTIAL"
    elif face_verified and liveness_verified and anti_cheat_passed:
        print("⚠️ PARTIAL SUCCESS")
        print("✅ Face Recognition: PASSED")
        print("✅ Liveness Check: PASSED")
        print("❌ PPE Detection: FAILED/INCOMPLETE")
        overall_result = "PARTIAL"
    elif face_verified and anti_cheat_passed:
        print("⚠️ PARTIAL SUCCESS")
        print("✅ Face Recognition: PASSED")
        print("❌ Liveness Check: FAILED/SKIPPED")
        print("❌ PPE Detection: SKIPPED (Face recognition required)")
        overall_result = "PARTIAL"
    else:
        print("❌ VERIFICATION FAILED")
        print("❌ Access denied")
        overall_result = "FAILED"
    
    print(f"📊 Final Match: {match_percent}%")
    print(f"⚠️ PPE Status:")
    print(f"   🪖 Helmet: {'✅ VERIFIED' if ppe_completed else '❌ NOT DETECTED'}")
    print(f"   � Glasses: {'✅ VERIFIED' if glasses_completed else '❌ NOT DETECTED'}")
    print(f"   😷 Mask: {'✅ VERIFIED' if mask_completed else '❌ NOT DETECTED'}")
    print(f"   �🧤 Gloves: {'✅ VERIFIED' if gloves_completed else '❌ NOT DETECTED'}")
    print(f"   👕 Vest: {'✅ VERIFIED' if vest_completed else '❌ NOT DETECTED'}")
    print(f"   🥾 Boots: {'✅ VERIFIED' if boots_completed else '❌ NOT DETECTED'}")
    
    # Auto-log results
    print(f"\n📝 Auto-logging verification results...")
    log_verification_result(face_verified, match_percent, liveness_verified, 
                          reference_path, threshold, anti_cheat_passed, overall_result, 
                          ppe_completed, glasses_completed, mask_completed, gloves_completed, vest_completed, boots_completed, matched_reference, vest_number)
    
    # Clean up temporary file
    try:
        if captured_path and os.path.exists(captured_path):
            os.remove(captured_path)
            print(f"🧹 Temporary file cleaned up")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up temporary file: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
