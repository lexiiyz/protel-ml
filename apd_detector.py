# ai-service/apd_detector.py
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np

class APDSystem:
    def __init__(self):
        print("⏳ Loading Models...")
        self.apdModel = YOLO("bestv2Revised.pt") 
        self.personModel = YOLO('yolov8n.pt') 
        
        self.ocrReader = easyocr.Reader(['en'], gpu=True) 
        self.REQUIRED_APD = {
            0: "Helmet",
            2: "Vest",
            6: "Boot",
            5: "Glove",   
        }
        print("✅ Models Loaded")

    def detect(self, frame):
        persons = self.personModel(frame, classes=0, conf=0.5, verbose=False)
        results = []

        # Jika tidak ada orang, return kosong biar cepat
        if not persons[0].boxes:
            return []

        for result in persons[0].boxes:
            box = result.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, box)

            # Validasi koordinat crop biar gak error
            h_img, w_img, _ = frame.shape
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(h_img, y2), min(w_img, x2)

            # Crop Person
            crop_img = frame[y1:y2, x1:x2]
            
            if crop_img.size == 0: continue

            apd_results = self.apdModel(crop_img, conf=0.25, verbose=False)[0]
            
            detected_ids = []
            if apd_results.boxes:
                detected_ids = apd_results.boxes.cls.tolist()
                detected_ids = list(map(int, detected_ids))

            # 3. Cek Kelengkapan
            missing = []
            is_compliant = True
            
            for req_id, req_name in self.REQUIRED_APD.items():
                if req_id not in detected_ids:
                    missing.append(req_name)
                    is_compliant = False

            ocr_text = None

            has_vest = 2 in detected_ids

            if not is_compliant and has_vest:
                try:
                    crop_ocr = cv2.resize(crop_img, (0,0), fx=0.5, fy=0.5) 
                    
                    ocr_res = self.ocrReader.readtext(crop_ocr, allowlist='0123456789', detail=1)
                    # Filter confidence > 0.4 biar gak baca sampah
                    valid_ocr = [res for res in ocr_res if res[2] > 0.4]
                    
                    if valid_ocr:
                        # Ambil yang paling confident
                        ocr_text = max(valid_ocr, key=lambda x: x[2])[1]
                except Exception as e:
                    print(f"OCR Error: {e}")

            results.append({
                "box": [x1, y1, x2, y2], 
                "is_compliant": is_compliant,
                "missing": missing,
                "ocr_id": ocr_text,
                # Debugging: kirim apa aja yang kedetect
                "detected_debug": detected_ids 
            })

        return results