from pathlib import Path
import shutil
import tempfile
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, FileResponse

from backend.detector import model, get_class_names, detect_image
from backend.safety_rules import detections_to_incidents
from backend.incident_store import initialize, add_incidents, list_incidents, update_status

app = FastAPI(
    title="SafeSite AI",
    description="YOLO11 industrial safety monitoring demo. Alerts require human verification.",
    version="1.0.0"
)
ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)
initialize()


@app.get("/")
def home():
    return {"project": "SafeSite AI", "message": "Backend is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/classes")
def classes():
    return {"classes": get_class_names()}


@app.post("/detect/image")
async def image_detection(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=0.99)
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(400, "Upload JPG, JPEG, PNG, WEBP or BMP.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not read image.")

    result = detect_image(image, confidence)
    annotated = result.plot()
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(500, "Could not encode annotated image.")

    incidents = detections_to_incidents(result, file.filename or "uploaded-image")
    add_incidents(incidents)

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Detection-Count": str(len(result.boxes) if result.boxes is not None else 0),
            "X-Incident-Count": str(len(incidents))
        }
    )


@app.post("/detect/video")
async def video_detection(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=0.99)
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "Upload MP4, AVI, MOV or MKV.")

    job_dir = Path(tempfile.mkdtemp(prefix="safesite_"))
    input_path = job_dir / f"input{suffix}"
    output_path = job_dir / "annotated.mp4"
    cap = writer = None

    try:
        with input_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise HTTPException(400, "Could not open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise HTTPException(400, "Invalid video dimensions.")

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height)
        )
        if not writer.isOpened():
            raise HTTPException(500, "Could not create output video.")

        frame_count = 0
        recorded_classes = set()
        incident_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detect_image(frame, confidence)
            writer.write(result.plot())
            frame_count += 1

            frame_incidents = detections_to_incidents(result, file.filename or "uploaded-video")
            new_incidents = []
            for incident in frame_incidents:
                if incident["class_name"] not in recorded_classes:
                    recorded_classes.add(incident["class_name"])
                    new_incidents.append(incident)
            if new_incidents:
                add_incidents(new_incidents)
                incident_count += len(new_incidents)

        cap.release()
        cap = None
        writer.release()
        writer = None

        if frame_count == 0 or not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(500, "No video frames processed.")

        final_path = EVIDENCE_DIR / f"safesite_{uuid.uuid4().hex}.mp4"
        shutil.copy2(output_path, final_path)
        return FileResponse(
            path=str(final_path), media_type="video/mp4",
            filename="safesite_annotated.mp4",
            headers={
                "X-Frames-Processed": str(frame_count),
                "X-Incident-Count": str(incident_count)
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Video processing failed: {exc}")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/incidents")
def incidents(limit: int = Query(500, ge=1, le=5000)):
    return {"incidents": list_incidents(limit)}


@app.patch("/incidents/{incident_id}")
def review_incident(incident_id: int, status: str = Query(...)):
    try:
        updated = update_status(incident_id, status)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not updated:
        raise HTTPException(404, "Incident not found.")
    return {"updated": True, "id": incident_id, "status": status}


@app.get("/stats")
def stats():
    items = list_incidents(5000)
    types = sorted({x["incident_type"] for x in items})
    return {
        "total_incidents": len(items),
        "high_priority": sum(x["severity"] == "High" for x in items),
        "needs_review": sum(x["review_status"] == "Needs review" for x in items),
        "incident_types": {t: sum(x["incident_type"] == t for x in items) for t in types}
    }
