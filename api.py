import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import base64
import numpy as np
import cv2
import face_recognition
import os
import sys
from apd_detector import APDSystem

# --- LOAD MODULE LOCAL ---
# Pastikan auto_verify.py ada di folder yang sama
try:
    import auto_verify as ao
    print("✅ Auto Verify Module Loaded")
except ImportError as e:
    print(f"❌ Failed to load auto_verify: {e}")
    ao = None

# --- INISIALISASI APP ---
app = FastAPI()
system = APDSystem()

# --- LOAD MODEL (DILAKUKAN SEKALI SAJA SAAT STARTUP) ---
print("⏳ Loading Models... (Please Wait)")

# 1. Load Liveness Predictor
if ao and hasattr(ao, 'LivenessDetector'):
    # Init dummy detector untuk trigger load dlib predictor ke memori
    _ = ao.LivenessDetector() 
    print("✅ Liveness Models Loaded")

# 2. Load YOLO & OCR (Trigger load awal)
if ao:
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    if hasattr(ao, 'detect_ppe_in_frame'):
        _ = ao.detect_ppe_in_frame(dummy_frame) # Pancing load YOLO
        print("✅ YOLO Model Loaded")
    if hasattr(ao, 'detect_vest_number_ocr'):
        # OCR biasanya lazy load, kita pancing dulu
        pass 

print("🚀 AI SERVER READY!")

# --- HELPER FUNCTIONS ---
def load_image_from_b64(b64str):
    try:
        if "," in b64str: b64str = b64str.split(",")[1]
        b = base64.b64decode(b64str)
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except: return None

def safe_face_encodings(img):
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes: return []
        return face_recognition.face_encodings(rgb, boxes)
    except: return []

# --- DATA STRUCTURE ---
class VerifyRequest(BaseModel):
    frames: List[str]          # List of Base64 strings
    refs: Dict[str, str]       # Dict { "id_pekerja": "base64_foto" }
    mode: str = "face"         # "face" atau "ppe"

class ImagePayload(BaseModel):
    image: str

def b64_to_img(uri):
    try:
        if "," in uri: uri = uri.split(",")[1]
        arr = np.frombuffer(base64.b64decode(uri), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except: return None

# --- LIVENESS ENDPOINT ---
@app.post("/verify_liveness")
async def verify_liveness_endpoint(req: VerifyRequest):
    out = {"success": False, "liveness_passed": False, "blinks": 0, "error": None}
    
    if not (ao and hasattr(ao, 'LivenessDetector')):
        out["error"] = "LivenessDetector module not loaded"
        return out
        
    try:
        frames = [img for b64 in req.frames if (img := load_image_from_b64(b64)) is not None]
        if not frames:
            out["error"] = "No valid frames provided"
            return out

        ld = ao.LivenessDetector()
        
        # We need a predictable challenge to validate against
        challenge = ld.generate_blink_challenge()
        
        for f in frames:
            # detect_blink_with_storage internal updates ld.total_blinks and ld.blink_timestamps
            ld.detect_blink_with_storage(f)
            
        timing_valid = ld.validate_human_timing()
        blinks_count = ld.total_blinks
        
        # Our frontend captures ~5.3 seconds of frames (800ms intervals? No, frontend burst captures faster. It's up to timing).
        # Wait, frontend `captureBurstFrames` captures locally and sends an array.
        
        out["blinks"] = blinks_count
        out["liveness_passed"] = (blinks_count >= 1) # Require at least 1 blink for liveness. 
        out["timing_valid"] = timing_valid
        out["success"] = True
        
        return out
    except Exception as e:
        print(f"❌ API Liveness Error: {e}")
        out["error"] = str(e)
        return out

# --- ENDPOINT UTAMA ---
@app.post("/verify")
async def verify_endpoint(req: VerifyRequest):
    out = {
        "success": False, 
        "face": {"match": False, "best_match_key": None, "best_match_percent": 0, "probe_has_face": False},
        "ppe": {
            "helmet_detected": False, "vest_detected": False,
            "gloves_detected": False, "boots_detected": False,
            "predictions": []
        },
        "vest_ocr": {"vest_number": None}
    }

    try:
        # 1. Decode Frames
        frames = [img for b64 in req.frames if (img := load_image_from_b64(b64)) is not None]
        if not frames:
            return out # Return empty success false

        best_frame = frames[-1]

        # 2. Liveness (Pilih Frame Terbaik)
        if ao and hasattr(ao, 'LivenessDetector'):
            try:
                ld = ao.LivenessDetector()
                for f in frames: ld.detect_blink_with_storage(f)
                sel = ld.select_best_frame_from_liveness_period()
                if sel and sel[0] is not None: best_frame = sel[0]
            except: pass

        # 3. MODE: FACE RECOGNITION
        if req.mode == "face":
            probe_encs = safe_face_encodings(best_frame)
            if probe_encs:
                out["face"]["probe_has_face"] = True
                probe_enc = probe_encs[0]
                best_score, best_name = 0.0, None
                
                # Loop Reference (Dari Request)
                for name, b64 in req.refs.items():
                    ref_img = load_image_from_b64(b64)
                    if ref_img is None: continue
                    ref_encs = safe_face_encodings(ref_img)
                    if not ref_encs: continue
                    
                    dist = face_recognition.face_distance([ref_encs[0]], probe_enc)[0]
                    match_percent = max(0.0, (1.0 - dist) * 100.0)
                    
                    if match_percent > best_score:
                        best_score = match_percent
                        best_name = name
                
                out["face"]["best_match_key"] = best_name
                out["face"]["best_match_percent"] = round(best_score, 1)
                out["face"]["match"] = bool(best_score >= 40.0)

        # 4. MODE: PPE & OCR
        if req.mode == "ppe" and ao:
            # YOLO
            try:
                if hasattr(ao, 'detect_ppe_in_frame'):
                    ppe = ao.detect_ppe_in_frame(best_frame)
                    out["ppe"]["predictions"] = ppe.get('predictions', [])
                    
                    for p in ppe.get('predictions', []):
                        cls = str(p.get('class', '')).lower()
                        if cls in ['0', 'helmet', 'hardhat']: out["ppe"]["helmet_detected"] = True
                        if cls in ['3', 'vest']: out["ppe"]["vest_detected"] = True
                        if cls in ['8', 'gloves']: out["ppe"]["gloves_detected"] = True
                        if cls in ['9', 'boots', 'shoes']: out["ppe"]["boots_detected"] = True
            except: pass

            # OCR
            try:
                if hasattr(ao, 'detect_vest_number_ocr'):
                    ocr = ao.detect_vest_number_ocr(best_frame)
                    out["vest_ocr"]["vest_number"] = ocr.get("vest_number")
            except: pass

        out["success"] = True
        return out

    except Exception as e:
        print(f"❌ API Error: {e}")
        return out
    
@app.post("/scan")
def scan_endpoint(req: ImagePayload):
    img = b64_to_img(req.image)
    if img is None: return {"success": False}
    
    # Panggil logic deteksi
    detections = system.detect(img)
    
    return {"success": True, "detections": detections}

if __name__ == "__main__":
    # Jalan di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)