"""Torch-free helpers: class names, Grad-CAM region zones, templated 'reason' text."""
import numpy as np

CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

CLASS_FINDINGS = {
    "NonDemented": "no notable atrophy pattern",
    "VeryMildDemented": "subtle, early structural changes",
    "MildDemented": "moderate structural changes, including some enlargement of the ventricles",
    "ModerateDemented": "pronounced atrophy, including ventricular enlargement and reduced tissue volume",
}


def detect_peak_region(grayscale_cam, center_frac=0.35, mid_frac=0.7):
    """Concentric zones around the image centre -> which zone holds the CAM peak."""
    h, w = grayscale_cam.shape
    cy, cx = h / 2, w / 2
    py, px = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    r = np.sqrt((py - cy) ** 2 + (px - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
    if r <= center_frac:
        return "ventricular / central", "the central region near the ventricles"
    if r <= mid_frac:
        return "hippocampal / temporal", "the medial temporal area associated with the hippocampus"
    return "cortical", "the outer cortical surface"


def confidence_band(conf):
    return "high" if conf >= 0.85 else "moderate" if conf >= 0.6 else "low"


def generate_explanation(pred_class, confidence, region_note):
    finding = CLASS_FINDINGS[pred_class]
    conf_phrase = {
        "high": "with high confidence",
        "moderate": "with moderate confidence",
        "low": "with relatively low confidence, so this prediction should be treated cautiously",
    }[confidence_band(confidence)]
    return (
        f"The model classified this scan as **{pred_class}** {conf_phrase} "
        f"({confidence * 100:.1f}%) because its attention was concentrated in {region_note}. "
        f"This is consistent with {finding}, which is a pattern typically seen in {pred_class.lower()} cases."
    )
