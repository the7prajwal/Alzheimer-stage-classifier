# 🧠 Alzheimer's Stage Classifier

Deep learning pipeline that classifies brain MRI scans into 4 Alzheimer's stages — **NonDemented**, **VeryMildDemented**, **MildDemented**, **ModerateDemented** — using transfer learning on ResNet18, with Grad-CAM visual explanations and a deployed Streamlit demo on Hugging Face Spaces.

Senior Design Project (SDP_CISHA_05).

🔗 **Live demo:** [Hugging Face Space](https://huggingface.co/spaces/the7prajwal/alzheimer-stage-demo)
🔗 **Model card:** [the7prajwal/alzheimer-stage-resnet18](https://huggingface.co/the7prajwal/alzheimer-stage-resnet18)

---

## Overview

The project follows a 6-stage pipeline:

1. **Dataset addition** — Kaggle Alzheimer's MRI dataset (4 classes)
2. **Preprocessing** — resizing, normalization, offline augmentation for the underrepresented `ModerateDemented` class
3. **Model training** — ResNet18 fine-tuned in two phases, best weights saved
4. **Result analysis** — confusion matrix, per-class F1, Grad-CAM
5. **Deployment** — model pushed to Hugging Face Hub
6. **Demo app** — Streamlit UI with live prediction + Grad-CAM overlay, hosted on HF Spaces

## Model

- **Architecture:** ResNet18 (ImageNet-pretrained)
- **Training:**
  - Phase 1 — backbone frozen, only the FC head trained
  - Phase 2 — `layer4` + FC unfrozen and fine-tuned
- **Loss:** Weighted CrossEntropyLoss to handle class imbalance
- **Augmentation:** Offline augmentation for `ModerateDemented`; horizontal flips deliberately excluded to preserve brain hemisphere asymmetry
- **Best result:** Val macro-F1 = **0.9775** (epoch 12)
- **Explainability:** Grad-CAM on `model.layer4[-1]` — attention concentrates on ventricles and periventricular cortex, and misclassifications cluster between clinically adjacent stages

> ⚠️ **Known limitation:** The source Kaggle dataset has documented train/test near-duplicate leakage, which likely inflates reported test accuracy. See the [model card](https://huggingface.co/the7prajwal/alzheimer-stage-resnet18) for details. This project is an academic demo and is **not for clinical use**.

## Repo contents

| File | Description |
|---|---|
| `AlzheimerStage.ipynb` | End-to-end notebook: data prep, training (both phases), evaluation, Grad-CAM, HF upload |
| `app.py` | Streamlit demo app — upload an MRI slice, get a predicted stage + Grad-CAM overlay |
| `requirements.txt` | Python dependencies for running the app |

## Running the demo locally

```bash
git clone https://github.com/the7prajwal/Alzheimer-stage-classifier.git
cd Alzheimer-stage-classifier
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the fine-tuned weights directly from the [Hugging Face model repo](https://huggingface.co/the7prajwal/alzheimer-stage-resnet18) on first run — no manual weight download needed.

## Tech stack

- PyTorch, torchvision (ResNet18)
- pytorch-grad-cam
- Streamlit
- Hugging Face Hub + Spaces

## Related

- Prior SDP-style project: [sentiment-analysis-BERT-NLTK](https://github.com/the7prajwal/sentiment-analysis-BERT-NLTK)

## Disclaimer

This is a student senior design project for educational purposes only. It is **not a medical device**, has not been clinically validated, and must not be used for real diagnosis or treatment decisions.
