import streamlit as st
import numpy as np
from PIL import Image

from models.registry import MODEL_REGISTRY
from utils.visualization import draw_results, CLASS_NAMES

st.set_page_config(page_title="Garbage Detector", page_icon="♻️", layout="wide")
st.title("♻️ Garbage Detection")
st.caption("Upload an image and select a model to detect waste items.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox("Model", list(MODEL_REGISTRY.keys()))
    selected_cls = MODEL_REGISTRY[model_name]
    st.caption(selected_cls.description)

    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.subheader("How to add a new model")
    st.markdown(
        "1. Create `models/your_model.py` and subclass `BaseDetector`\n"
        "2. Set `name` and `description` class attributes\n"
        "3. Implement `load()` and `predict()`\n"
        "4. Add `from models import your_model` in `models/registry.py`"
    )

# ── Model loading (cached per model name) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def get_detector(name: str):
    detector = MODEL_REGISTRY[name]()
    detector.load()
    return detector

detector = get_detector(model_name)

# ── File uploader ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

pil_image = Image.open(uploaded).convert("RGB")
image_np = np.array(pil_image)

# ── Run inference ──────────────────────────────────────────────────────────────
with st.spinner("Running detection…"):
    result = detector.predict(image_np, conf_threshold)

boxes  = result["boxes"]
labels = result["labels"]
scores = result["scores"]
masks  = result["masks"]

annotated = draw_results(image_np, boxes, labels, scores, masks)

# ── Display images ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(pil_image, use_container_width=True)
with col2:
    st.subheader(f"Detections ({len(boxes)} found)")
    st.image(annotated, use_container_width=True)

# ── Results table ──────────────────────────────────────────────────────────────
if boxes:
    st.subheader("Detection Details")

    # Per-class count summary
    from collections import Counter
    counts = Counter(CLASS_NAMES[max(0, min(l, len(CLASS_NAMES)-1))] for l in labels)
    summary_cols = st.columns(len(counts))
    for col, (cls, cnt) in zip(summary_cols, sorted(counts.items())):
        col.metric(cls, cnt)

    st.divider()

    # Full table
    rows = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        label = max(0, min(label, len(CLASS_NAMES) - 1))
        x1, y1, x2, y2 = [int(v) for v in box]
        rows.append({
            "#": i + 1,
            "Class": CLASS_NAMES[label],
            "Confidence": f"{score:.3f}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Width": x2 - x1, "Height": y2 - y1,
        })

    st.dataframe(rows, use_container_width=True)
else:
    st.warning(f"No objects detected above {conf_threshold:.0%} confidence. Try lowering the threshold.")
