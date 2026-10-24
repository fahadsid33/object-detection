import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SafeSite AI", page_icon="🦺", layout="wide")
API_URL = st.sidebar.text_input("Backend API URL", "http://127.0.0.1:8000").rstrip("/")

st.title("🦺 SafeSite AI")
st.caption("YOLO11 Industrial Safety Monitoring — AI alerts require human verification.")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Image Detection", "Video Detection", "Incident History", "Model Classes"]
)


def api_get(path):
    return requests.get(f"{API_URL}{path}", timeout=20)


def upload_request(path, uploaded, confidence, timeout=900):
    return requests.post(
        f"{API_URL}{path}",
        params={"confidence": confidence},
        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
        timeout=timeout
    )


if page == "Overview":
    st.subheader("System overview")
    try:
        health = api_get("/health")
        health.raise_for_status()
        st.success("FastAPI backend connected")
        response = api_get("/stats")
        response.raise_for_status()
        stats = response.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Possible incidents", stats["total_incidents"])
        c2.metric("High priority", stats["high_priority"])
        c3.metric("Needs review", stats["needs_review"])
        if stats["incident_types"]:
            st.subheader("Incident categories")
            st.bar_chart(pd.DataFrame({"Count": stats["incident_types"]}))
        else:
            st.info("No incidents yet. Upload a test image or video.")
    except requests.RequestException as exc:
        st.error(f"Backend is not reachable: {exc}")
        st.info("Start FastAPI in a separate terminal first.")

elif page == "Image Detection":
    st.subheader("Image detection")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded and st.button("Run detection", type="primary"):
        with st.spinner("Running YOLO11..."):
            try:
                response = upload_request("/detect/image", uploaded, confidence, 180)
                response.raise_for_status()
                st.image(response.content, caption="Annotated result", use_container_width=True)
                st.write("Objects detected:", response.headers.get("X-Detection-Count", "unknown"))
                st.write("Possible incidents recorded:", response.headers.get("X-Incident-Count", "unknown"))
                st.download_button("Download annotated image", response.content, "safesite_result.jpg", "image/jpeg")
            except requests.RequestException as exc:
                st.error(f"Detection failed: {exc}")
                if getattr(exc, "response", None) is not None:
                    st.code(exc.response.text)

elif page == "Video Detection":
    st.subheader("Video detection")
    st.warning("Video processing may take time depending on video size and hardware.")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="video_conf")
    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded and st.button("Analyze video", type="primary"):
        with st.spinner("Processing video frames..."):
            try:
                response = upload_request("/detect/video", uploaded, confidence, 900)
                response.raise_for_status()
                video_bytes = response.content
                st.video(video_bytes)
                st.write("Frames processed:", response.headers.get("X-Frames-Processed", "see backend"))
                st.write("New possible incidents:", response.headers.get("X-Incident-Count", "see incident history"))
                st.download_button("Download annotated video", video_bytes, "safesite_annotated.mp4", "video/mp4")
            except requests.RequestException as exc:
                st.error(f"Video processing failed: {exc}")
                if getattr(exc, "response", None) is not None:
                    st.code(exc.response.text)

elif page == "Incident History":
    st.subheader("Incident history")
    try:
        response = api_get("/incidents")
        response.raise_for_status()
        rows = response.json().get("incidents", [])
        if not rows:
            st.info("No possible incidents recorded yet.")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "safesite_incidents.csv", "text/csv")
            st.markdown("### Review an incident")
            incident_id = st.selectbox("Incident ID", [int(r["id"]) for r in rows])
            status = st.selectbox("Status", ["Reviewed", "Dismissed", "Needs review"])
            if st.button("Save review status"):
                update = requests.patch(
                    f"{API_URL}/incidents/{incident_id}",
                    params={"status": status},
                    timeout=20
                )
                update.raise_for_status()
                st.success("Status updated. Reopen or refresh this page to see it.")
    except requests.RequestException as exc:
        st.error(f"Could not load incidents: {exc}")

elif page == "Model Classes":
    st.subheader("Classes recognized by the trained model")
    try:
        response = api_get("/classes")
        response.raise_for_status()
        st.json(response.json())
        st.caption("Only classes listed here can be detected by this model. Check class labels before interpreting PPE alerts.")
    except requests.RequestException as exc:
        st.error(f"Could not load classes: {exc}")
