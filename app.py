import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================================================
# Config
# ============================================================
HF_REPO_ID = "the7prajwal/alzheimer-stage-resnet18"
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Alzheimer's Stage Detection", layout="centered")

# ============================================================
# Load model (cached so it only downloads/loads once per session)
# ============================================================
@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="pytorch_model.bin")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def denormalize(tensor_img):
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)

# ============================================================
# UI
# ============================================================
st.title("🧠 Alzheimer's Stage Detection from Brain MRI")
st.caption("SDP_CISHA_05 — Educational demo. Not for clinical use.")

st.warning(
    "⚠️ This is an academic project, not a medical device. "
    "It has not been clinically validated and must not be used for real diagnosis."
)

uploaded_file = st.file_uploader("Upload a brain MRI slice (axial view)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    rgb_img = denormalize(input_tensor[0])
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
    with col2:
        st.image(cam_overlay, caption="Grad-CAM (model attention)", use_container_width=True)

    st.subheader(f"Prediction: **{CLASSES[pred_idx]}**")
    st.write(f"Confidence: {probs[pred_idx]*100:.2f}%")

    st.bar_chart({cls: float(p) for cls, p in zip(CLASSES, probs)})

    st.info(
        "Grad-CAM highlights regions (e.g. ventricles, cortex) that most influenced this prediction. "
        "It shows *where* the model looked, not a diagnosis."
    )
else:
    st.info("Upload an MRI slice above to see a prediction and Grad-CAM explanation.")

st.markdown("---")
st.caption(
    f"Model: [ResNet18 fine-tuned on Kaggle Alzheimer's Dataset]"
    f"(https://huggingface.co/{HF_REPO_ID}) · "
    "Test metrics may be inflated due to known train/test overlap in the source dataset — see model card for details."
)