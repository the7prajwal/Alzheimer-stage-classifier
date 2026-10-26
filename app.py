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
# Sidebar: About Alzheimer's + the 4 stages
# ============================================================
with st.sidebar:
    st.header("🧠 About Alzheimer's Disease")
    st.write(
        "Alzheimer's is a progressive brain disorder that gradually damages "
        "memory and thinking skills. Over time it causes the brain's tissue to "
        "shrink (atrophy) and its ventricles (fluid-filled cavities) to enlarge, "
        "which is what this model looks for in MRI scans."
    )

    st.subheader("The 4 Stages")

    st.markdown("**1. NonDemented**")
    st.caption(
        "No noticeable memory loss or structural changes. Brain scans typically "
        "look normal for the person's age."
    )

    st.markdown("**2. VeryMildDemented**")
    st.caption(
        "Early, subtle changes — occasional forgetfulness (e.g. misplacing "
        "objects, minor word-finding issues) that are often mistaken for normal aging."
    )

    st.markdown("**3. MildDemented**")
    st.caption(
        "Clearer memory and cognitive difficulties that others start to notice — "
        "trouble with planning, recent events, or handling finances. Scans may show "
        "early ventricular enlargement."
    )

    st.markdown("**4. ModerateDemented**")
    st.caption(
        "Significant memory loss and confusion affecting daily life, requiring "
        "assistance with routine tasks. Scans often show pronounced atrophy and "
        "ventricular enlargement."
    )

    st.markdown("---")
    st.caption(
        "This is a simplified, educational summary for an academic project — "
        "not medical guidance. Consult a healthcare professional for real diagnosis."
    )

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
# Explainability: region detection + templated explanation (Option 1)
# ============================================================
def detect_peak_region(grayscale_cam, center_frac=0.35, mid_frac=0.7):
    """
    Divide the CAM into concentric zones and report which zone
    contains the peak activation.
      - center zone  -> "ventricular / central" region
      - mid zone     -> "hippocampal / temporal" region
      - outer zone   -> "cortical" region
    """
    h, w = grayscale_cam.shape
    cy, cx = h / 2, w / 2

    peak_idx = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    py, px = peak_idx

    # normalized radial distance of the peak from the center (0 = center, 1 = edge)
    max_r = np.sqrt(cy**2 + cx**2)
    r = np.sqrt((py - cy) ** 2 + (px - cx) ** 2) / max_r

    if r <= center_frac:
        region = "ventricular / central"
        region_note = "the central region near the ventricles"
    elif r <= mid_frac:
        region = "hippocampal / temporal"
        region_note = "the medial temporal area associated with the hippocampus"
    else:
        region = "cortical"
        region_note = "the outer cortical surface"

    return region, region_note


CLASS_FINDINGS = {
    "NonDemented": "no notable atrophy pattern",
    "VeryMildDemented": "subtle, early structural changes",
    "MildDemented": "moderate structural changes, including some enlargement of the ventricles",
    "ModerateDemented": "pronounced atrophy, including ventricular enlargement and reduced tissue volume",
}


def generate_explanation(pred_class, confidence, region, region_note):
    finding = CLASS_FINDINGS[pred_class]

    if confidence >= 0.85:
        conf_phrase = "with high confidence"
    elif confidence >= 0.6:
        conf_phrase = "with moderate confidence"
    else:
        conf_phrase = "with relatively low confidence, so this prediction should be treated cautiously"

    reason_text = (
        f"The model classified this scan as **{pred_class}** {conf_phrase} "
        f"({confidence*100:.1f}%) because its attention was concentrated in {region_note}. "
        f"This is consistent with {finding}, which is a pattern typically seen in {pred_class.lower()} cases."
    )
    return reason_text

# ============================================================
# UI
# ============================================================
st.title("🧠 Alzheimer's Stage Detection from Brain MRI")
st.caption("SDP_CISHA_04 — Educational demo. Not for clinical use.")

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
    region, region_note = detect_peak_region(grayscale_cam)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
    with col2:
        st.image(cam_overlay, caption="Grad-CAM (model attention)", use_container_width=True)

    st.subheader(f"Prediction: **{CLASSES[pred_idx]}**")
    st.write(f"Confidence: {probs[pred_idx]*100:.2f}%")

    st.bar_chart({cls: float(p) for cls, p in zip(CLASSES, probs)})

    explanation = generate_explanation(
        CLASSES[pred_idx], float(probs[pred_idx]), region, region_note
    )

    st.markdown(
        f"<p style='font-size:20px; margin-bottom:0;'>"
        f"<strong>Brain region contributing most:</strong> {region}</p>",
        unsafe_allow_html=True,
    )

    st.subheader("🔍 Reason for Detection")
    st.markdown(f"**Reason:** {explanation}")

    st.caption(
        "This explanation is generated from a rule-based mapping between the Grad-CAM "
        "attention zone and known anatomical patterns per stage. It describes *where* the "
        "model looked and what that typically corresponds to — it is not a diagnosis."
    )
else:
    st.info("Upload an MRI slice above to see a prediction and Grad-CAM explanation.")

st.markdown("---")
st.caption(
    f"Model: [ResNet18 fine-tuned on Kaggle Alzheimer's Dataset]"
    f"(https://huggingface.co/{HF_REPO_ID}) · "
    "Test metrics may be inflated due to known train/test overlap in the source dataset — see model card for details."
)