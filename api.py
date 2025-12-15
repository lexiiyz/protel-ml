import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import base64
import cv2
from uni_detector import UnifiedSystem

# Init App & System
app = FastAPI()
system = UnifiedSystem()

print("🚀 UNIFIED AI SERVER READY!")

# Models
class VerifyRequest(BaseModel):
    frames: List[str]
    refs: Dict[str, str]
    mode: str = "face" 

class ImagePayload(BaseModel):
    image: str
    cameraId: int = 0

# Helper
def b64_to_img(uri):
    try:
        if "," in uri: uri = uri.split(",")[1]
        arr = np.frombuffer(base64.b64decode(uri), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except: return None

# --- ABSENSI ENDPOINT ---
@app.post("/verify")
async def verify_endpoint(req: VerifyRequest):
    out = {
        "success": False, 
        "face": {"match": False, "best_match_key": None, "best_match_percent": 0, "probe_has_face": False},
        "ppe": {
            # Tambahkan default predictions list kosong
            "predictions": [], 
            "helmet_detected": False, "vest_detected": False, 
            "gloves_detected": False, "boots_detected": False
        },
        "vest_ocr": {"vest_number": None}
    }

    try:
        img = None
        for b64 in reversed(req.frames):
            img = b64_to_img(b64)
            if img is not None: break
        
        if img is None: return out

        # MODE: FACE
        if req.mode == "face":
            res = system.verify_face(img, req.refs)
            out["face"]["probe_has_face"] = res["has_face"]
            out["face"]["match"] = res["match"]
            out["face"]["best_match_key"] = res["name"]
            out["face"]["best_match_percent"] = res["percent"]

        # MODE: PPE (PURE ML)
        elif req.mode == "ppe":
            res = system.detect_ppe(img)
            
            # Mapping result unified ke output API
            out["ppe"]["helmet_detected"] = res["helmet_detected"]
            out["ppe"]["vest_detected"] = res["vest_detected"]
            out["ppe"]["gloves_detected"] = res["gloves_detected"]
            out["ppe"]["boots_detected"] = res["boots_detected"]
            
            # INI PENTING: Kirim list bounding box ke frontend
            out["ppe"]["predictions"] = res["predictions"]
            
            out["vest_ocr"]["vest_number"] = res["vest_number"]

        out["success"] = True
        return out

    except Exception as e:
        print(f"❌ API Error: {e}")
        return out
    
# --- MONITORING ENDPOINT (CCTV) ---
@app.post("/scan")
def scan_endpoint(req: ImagePayload):
    img = b64_to_img(req.image)
    if img is None: return {"success": False}
    
    # 1. Panggil Unified Detector
    res = system.detect_ppe(img)
    detections = []
    if res["found"]:
        is_compliant = (
            res["helmet_detected"] 
            and res["vest_detected"]
            and res["gloves_detected"]
            and res["boots_detected"]
        )
        
        missing = []
        if not res["helmet_detected"]: missing.append("Helmet")
        if not res["vest_detected"]: missing.append("Vest")
        if not res["gloves_detected"]: missing.append("Gloves")
        if not res["boots_detected"]: missing.append("Boots")
        
        # Kirim Box ASLI dari YOLO
        real_box = res["box"]
        
        detections.append({
            "box": real_box,
            "is_compliant": is_compliant,
            "missing": missing,
            "ocr_id": res["vest_number"]
        })
    
    return {"success": True, "detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)