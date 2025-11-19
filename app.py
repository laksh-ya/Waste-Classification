import streamlit as st
import warnings, os
warnings.filterwarnings("ignore")

from fastai.vision.all import *
from PIL import Image

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# ----------------------------------------------------
# CSS — MINIMAL, CLEAN, MODERN
# ----------------------------------------------------
st.markdown("""
<style>

body {
    background: #f5f5f5;
}

.block-container {
    max-width: 650px;
    margin: auto;
    padding-top: 2rem;
}

/* Title */
.title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #222;
    text-align: center;
    margin-bottom: 0.3rem;
}

/* Subtitle */
.subtitle {
    text-align:center;
    color:#666;
    font-size:1rem;
    margin-bottom:1.8rem;
}

/* Upload Box */
.upload-zone {
    border: 2px dashed #bfbfbf;
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    color: #777;
    font-size: 1rem;
    background: white;
    margin-bottom: 1rem;
}

/* Sample section */
.sample-title {
    text-align:center;
    color:#888;
    margin-top:1.4rem;
    margin-bottom:0.7rem;
    font-size:0.95rem;
}

.sample-img {
    border-radius: 12px;
    cursor: pointer;
    border: 1px solid #ddd;
    transition: 0.15s;
}

.sample-img:hover {
    transform: scale(1.05);
}

/* Button */
.stButton button {
    width: 100%;
    padding: 0.75rem;
    border-radius: 12px;
    background: #4CAF50;
    color: white;
    border:none;
    font-weight:600;
    transition: 0.15s;
}

.stButton button:hover {
    background: #45a047;
}

/* Result */
.result-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem;
    margin-top: 1.3rem;
    box-shadow: 0px 2px 15px rgba(0,0,0,0.1);
    text-align:center;
}

.result-label {
    font-size: 1.4rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return load_learner("model_on-4.pkl")

learn = load_model()

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.markdown("<h1 class='title'>♻ waste classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>fastai • pytorch • resnet18 • v4 model</p>", unsafe_allow_html=True)

# ----------------------------------------------------
# STATE
# ----------------------------------------------------
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# ----------------------------------------------------
# SAMPLE IMAGES (2×2 PANEL)
# ----------------------------------------------------
sample_dir = "example_images"
sample_imgs = []

if os.path.exists(sample_dir):
    for f in sorted(os.listdir(sample_dir))[:4]:   # EXACTLY 4 images
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            sample_imgs.append(os.path.join(sample_dir, f))

# ----------------------------------------------------
# UPLOADER
# ----------------------------------------------------
uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

# auto-load sample selection
if st.session_state.selected_image and uploaded is None:
    uploaded = st.session_state.selected_image

# ----------------------------------------------------
# SAMPLE GRID (COLLAPSIBLE + SMALL THUMBNAILS)
# ----------------------------------------------------
if sample_imgs:

    with st.expander("try sample images (tap to open)"):
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        grid = [col1, col2, col3, col4]

        for idx, path in enumerate(sample_imgs):
            with grid[idx]:

                # small uniform thumbnails
                img_obj = Image.open(path).resize((120, 120))
                st.image(img_obj, width=120)

                if st.button("use this", key=f"sample_{idx}"):
                    st.session_state.selected_image = path
                    st.rerun()
# ----------------------------------------------------
# PREVIEW + CLASSIFY
# ----------------------------------------------------
if uploaded:
    img = PILImage.create(uploaded)
    st.image(img, caption="preview", use_container_width=True)

    if st.button("classify image 🔍"):
        with st.spinner("analyzing…"):
            pred, idx, probs = learn.predict(img)
            conf = float(probs[idx]) * 100

        if str(pred).lower().startswith("b"):
            label = "Biodegradable 🌱"
            color = "#4CAF50"
        else:
            label = "Non-Biodegradable ♻️"
            color = "#E53935"

        st.markdown(
            f"""
            <div class='result-card'>
                <span class='result-label' style='color:{color};'>{label}</span>
                <p style='margin-top:0.6rem;color:#444;'>
                    confidence: <b>{conf:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.markdown("<div class='upload-zone'>drag & drop an image here<br>or tap to upload</div>", unsafe_allow_html=True)