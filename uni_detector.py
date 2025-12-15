import cv2
import numpy as np
import face_recognition
import base64
from ultralytics import YOLO
import easyocr

class UnifiedSystem:
    def __init__(self):
        print("⏳ Loading Models (PURE ML + OCR)...")
        self.apdModel = YOLO("bestv2Revised.pt") 
        self.personModel = YOLO('yolov8n.pt') 
        
        self.REQUIRED_APD = {
            0: "Helmet", 2: "Vest", 5: "Glove", 6: "Boot"
        }
        self.ocrReader = easyocr.Reader(['en'], gpu=True)
        print("✅ Unified Models Loaded")

    def load_image(self, b64str):
        try:
            if "," in b64str: b64str = b64str.split(",")[1]
            arr = np.frombuffer(base64.b64decode(b64str), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except: return None

    # --- FACE RECOGNITION (SAMA) ---
    def verify_face(self, frame, refs):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            probe_locs = face_recognition.face_locations(rgb, model="hog")
            if not probe_locs: return {"match": False, "percent": 0, "name": None, "has_face": False}

            probe_enc = face_recognition.face_encodings(rgb, probe_locs)[0]
            best_score = 0.0
            best_name = None

            for name, b64_ref in refs.items():
                ref_img = self.load_image(b64_ref)
                if ref_img is None: continue
                
                ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_locs = face_recognition.face_locations(ref_rgb, model="hog")
                if not ref_locs: continue
                
                ref_enc = face_recognition.face_encodings(ref_rgb, ref_locs)[0]
                dist = face_recognition.face_distance([ref_enc], probe_enc)[0]
                match_percent = max(0.0, (1.0 - dist) * 100.0)
                if match_percent > best_score:
                    best_score = match_percent
                    best_name = name

            return {"match": bool(best_score >= 40.0), "percent": float(round(best_score, 1)), "name": best_name, "has_face": True}
        except Exception as e:
            print(f"❌ Face Error: {e}")
            return {"match": False, "percent": 0, "name": None, "has_face": False}

    # --- APD DETECTION (TUNED THRESHOLD) ---
    def detect_ppe(self, frame):
        h_img, w_img, _ = frame.shape
        persons = self.personModel(frame, classes=0, conf=0.5, verbose=False)
        
        final_result = {
            "found": False, "box": [],
            "helmet_detected": False, "vest_detected": False, 
            "gloves_detected": False, "boots_detected": False, 
            "vest_number": None, "predictions": [] 
        }

        if not persons[0].boxes: return final_result

        person_box = persons[0].boxes[0].xyxy.cpu().numpy()[0]
        p_x1, p_y1, p_x2, p_y2 = map(int, person_box)
        final_result["found"] = True
        final_result["box"] = [p_x1, p_y1, p_x2, p_y2]

        crop = frame[max(0,p_y1):min(h_img,p_y2), max(0,p_x1):min(w_img,p_x2)]
        if crop.size == 0: return final_result

        # UBAH DISINI: Turunkan confidence ke 0.20 biar Gloves/Boots yang samar kebaca
        apd_results = self.apdModel(crop, conf=0.20, verbose=False)[0]
        
        if apd_results.boxes:
            for box in apd_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                c_x1, c_y1, c_x2, c_y2 = box.xyxy[0].tolist()
                g_x1, g_y1, g_x2, g_y2 = c_x1 + p_x1, c_y1 + p_y1, c_x2 + p_x1, c_y2 + p_y1
                
                norm_x = ((g_x1 + g_x2) / 2) / w_img
                norm_y = ((g_y1 + g_y2) / 2) / h_img
                norm_w = (g_x2 - g_x1) / w_img
                norm_h = (g_y2 - g_y1) / h_img
                
                class_name = ""
                # Logic Threshold per Item
                if cls_id == 0 and conf > 0.5:
                    final_result["helmet_detected"] = True
                    class_name = "helmet"
                elif cls_id == 2 and conf > 0.4: 
                    final_result["vest_detected"] = True
                    class_name = "vest"
                elif cls_id == 5 and conf > 0.20: 
                    final_result["gloves_detected"] = True
                    class_name = "gloves"
                elif cls_id == 6 and conf > 0.20:
                    final_result["boots_detected"] = True
                    class_name = "boots"
                
                if class_name:
                    final_result["predictions"].append({
                        "class": class_name, "confidence": conf,
                        "box": {"x": norm_x, "y": norm_y, "w": norm_w, "h": norm_h}
                    })

        # OCR Logic
        if final_result["vest_detected"]:
            try:
                ocr_res = self.ocrReader.readtext(crop, allowlist='0123456789', detail=1)
                valid = [res for res in ocr_res if res[2] > 0.4]
                if valid:
                    final_result["vest_number"] = max(valid, key=lambda x: x[2])[1]
            except: pass

        return final_result