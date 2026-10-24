from datetime import datetime, timezone

# Basic starter rules: edit these labels to match your model's actual class names.
PPE_RISK_LABELS = {
    "no-hardhat": ("Possible missing hardhat", "High"),
    "no_hardhat": ("Possible missing hardhat", "High"),
    "no-safety-vest": ("Possible missing safety vest", "Medium"),
    "no_safety_vest": ("Possible missing safety vest", "Medium"),
    "no-vest": ("Possible missing safety vest", "Medium"),
    "no_vest": ("Possible missing safety vest", "Medium"),
}


def detections_to_incidents(result, source_name: str):
    incidents = []
    names = result.names or {}
    if result.boxes is None:
        return incidents

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = str(names.get(class_id, class_id))
        key_dash = label.strip().lower().replace(" ", "-")
        key_underscore = label.strip().lower().replace(" ", "_")
        match = PPE_RISK_LABELS.get(key_dash) or PPE_RISK_LABELS.get(key_underscore)
        if match:
            incident_type, severity = match
            incidents.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": source_name,
                "incident_type": incident_type,
                "severity": severity,
                "confidence": round(confidence, 4),
                "class_name": label,
                "review_status": "Needs review",
            })
    return incidents
