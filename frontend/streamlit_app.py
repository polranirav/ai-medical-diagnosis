import streamlit as st
import requests
from PIL import Image
import io
import json
import os

API_BASE = os.environ.get('API_BASE', 'http://127.0.0.1:8001')
PRED_ENDPOINT = f"{API_BASE}/predict"
PROBA_ENDPOINT = f"{API_BASE}/predict_proba"
MODEL_INFO_ENDPOINT = f"{API_BASE}/model_info"

st.set_page_config(
    page_title="AI Chest X-Ray Diagnosis",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem;}
    .result-box {padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
    .normal {background-color: #d4edda; border-left: 6px solid #28a745;}
    .pneumonia {background-color: #f8d7da; border-left: 6px solid #dc3545;}
    .footer-note {font-size: 0.8rem; color: #666; margin-top: 2rem; text-align:center;}
</style>
""", unsafe_allow_html=True)

def get_model_info():
    try:
        r = requests.get(MODEL_INFO_ENDPOINT, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        return {}
    return {}

def call_endpoint(url, file_bytes):
    try:
        resp = requests.post(url, files={'file': file_bytes}, timeout=30)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

st.markdown('<h1 class="main-header">🩺 AI Chest X-Ray Diagnosis System</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("About This Tool")
    st.write("""This AI system analyzes frontal chest X-rays for pneumonia likelihood.\n\n**Performance (recent run)**\n- Accuracy: 83.2%\n- ROC-AUC: 96.2%\n- Pneumonia Recall: ~100%\n\n**Disclaimer:** For research/education only. Not a medical device.""")
    info = get_model_info()
    if info:
        st.subheader("Model Info")
        st.write(f"Arch: {info.get('architecture')}")
        st.write(f"Temp: {info.get('temperature')}")
        st.write(f"Calibrated: {info.get('calibrated')}")
        st.write(f"Threshold: {info.get('threshold')}")
        if info.get('checkpoint_sha256'):
            st.write(f"CKPT Hash: `{info['checkpoint_sha256']}`")

col1, col2 = st.columns([1,1])

with col1:
    st.header("📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["png","jpg","jpeg"])    
    analyze = st.button("🔍 Analyze X-Ray", type="primary", disabled=uploaded_file is None)

    if analyze and uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        file_bytes = uploaded_file.getvalue()
        with st.spinner("Contacting AI service..."):
            pred_json = call_endpoint(PRED_ENDPOINT, file_bytes)
            proba_json = call_endpoint(PROBA_ENDPOINT, file_bytes)
        st.session_state['pred_json'] = pred_json
        st.session_state['proba_json'] = proba_json

with col2:
    st.header("📋 Results")
    pred_json = st.session_state.get('pred_json')
    proba_json = st.session_state.get('proba_json')
    if not pred_json or not proba_json:
        st.info("Upload an image and click Analyze to see results.")
    else:
        if 'error' in pred_json or 'error' in proba_json:
            st.error(f"Error: {pred_json.get('error') or proba_json.get('error')}")
        else:
            probs = proba_json.get('probabilities', [0,0])
            classes = proba_json.get('classes', ["NORMAL","PNEUMONIA"])
            temp = proba_json.get('temperature')
            threshold = proba_json.get('threshold')
            pneumonia_prob = probs[1] if len(probs)>1 else 0
            predicted = proba_json.get('top_class', classes[int(pneumonia_prob>=threshold)])
            css_class = 'pneumonia' if predicted == 'PNEUMONIA' else 'normal'
            conf_display = pneumonia_prob if predicted=='PNEUMONIA' else probs[0]
            st.markdown(f"""
            <div class="result-box {css_class}">
              <h3>{'⚠️' if predicted=='PNEUMONIA' else '✅'} Prediction: {predicted}</h3>
              <p><strong>Confidence:</strong> {conf_display:.1%}</p>
              <p><strong>Temperature (T):</strong> {temp}</p>
              <p><strong>Decision Threshold:</strong> {threshold}</p>
            </div>
            """, unsafe_allow_html=True)
            st.subheader("Detailed Probabilities")
            for cls, p in zip(classes, probs):
                st.write(f"**{cls}**: {p:.1%}")
                st.progress(min(max(p,0.0),1.0))
            st.warning("This tool does not provide a medical diagnosis. Consult a radiologist for clinical decisions.")

st.markdown('<div class="footer-note">© 2025 AI Medical Diagnosis (Research Prototype)</div>', unsafe_allow_html=True)
