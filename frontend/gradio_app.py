import gradio as gr
import requests
from PIL import Image
import io, os

API_BASE = os.environ.get('API_BASE', 'http://127.0.0.1:8001')
PRED_ENDPOINT = f"{API_BASE}/predict"
PROBA_ENDPOINT = f"{API_BASE}/predict_proba"


def predict_xray(image: Image.Image):
    if image is None:
        return "No image provided"
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    files = {'file': img_bytes}
    try:
        r_pred = requests.post(PRED_ENDPOINT, files=files, timeout=30)
        r_proba = requests.post(PROBA_ENDPOINT, files=files, timeout=30)
        pred_js = r_pred.json()
        proba_js = r_proba.json()
    except Exception as e:
        return f"Request error: {e}"
    probs = proba_js.get('probabilities', [0,0])
    classes = proba_js.get('classes', ["NORMAL","PNEUMONIA"])
    threshold = proba_js.get('threshold', 0.5)
    pneumonia_prob = probs[1] if len(probs)>1 else 0
    predicted = proba_js.get('top_class', classes[int(pneumonia_prob>=threshold)])
    lines = [
        f"AI Diagnosis: {predicted}",
        f"Normal: {probs[0]:.2%}" if len(probs)>0 else "Normal: N/A",
        f"Pneumonia: {pneumonia_prob:.2%}",
        f"Threshold: {threshold}",
        "Disclaimer: Research use only."
    ]
    return '\n'.join(lines)


demo = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type='pil'),
    outputs=gr.Textbox(label='AI Analysis Results'),
    title='🩺 AI Chest X-Ray Diagnosis System',
    description='Upload a chest X-ray to get AI-driven probability estimates.',
    allow_flagging='never'
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=8502, share=False)
