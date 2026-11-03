"""ResNet18 + Grad-CAM wrapper. Returns everything the app needs from one call."""
import io
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torchvision import models, transforms

from explain import CLASSES, detect_peak_region, generate_explanation

HF_REPO_ID = "the7prajwal/alzheimer-stage-classifier-resnet18"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Classifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = hf_hub_download(repo_id=HF_REPO_ID, filename="pytorch_model.bin")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASSES))
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.to(self.device).eval()
        self.cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def predict(self, image: Image.Image) -> dict:
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        i = int(np.argmax(probs))
        grayscale_cam = self.cam(input_tensor=x, targets=None)[0]
        rgb = np.clip(x[0].cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN), 0, 1)
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, "PNG")
        region, region_note = detect_peak_region(grayscale_cam)
        conf = float(probs[i])
        return {
            "stage": CLASSES[i],
            "confidence": conf,
            "probabilities": {k: float(p) for k, p in zip(CLASSES, probs)},
            "region": region,
            "region_note": region_note,
            "gradcam_png": buf.getvalue(),
            "explanation": generate_explanation(CLASSES[i], conf, region_note),
        }
