import io
import os

import pandas as pd
import streamlit as st
from PIL import Image

import db
from explain import CLASSES
from pdf_utils import SECTIONS, pdf_path, save_pdf
from report_agent import (DISCLAIMER, MIN_CONFIDENCE, OLLAMA_MODEL, compute_trend, generate_report,
                          recommendations)
from explain import confidence_band

st.set_page_config(page_title="Alzheimer's Stage Detection & Reports", layout="centered")
db.init_db()


@st.cache_resource
def get_classifier():
    from model_utils import Classifier  # imported lazily so the DB/history pages load fast
    return Classifier()


# ============================================================
# Sidebar (same content as the original app)
# ============================================================
with st.sidebar:
    st.header("🧠 About Alzheimer's Disease")
    st.write(
        "Alzheimer's is a progressive brain disorder that gradually damages memory and thinking skills. "
        "Over time it causes the brain's tissue to shrink (atrophy) and its ventricles (fluid-filled "
        "cavities) to enlarge, which is what this model looks for in MRI scans."
    )
    st.subheader("The 4 Stages")
    for title, text in [
        ("1. NonDemented", "No noticeable memory loss or structural changes. Brain scans typically look normal for the person's age."),
        ("2. VeryMildDemented", "Early, subtle changes — occasional forgetfulness that is often mistaken for normal aging."),
        ("3. MildDemented", "Clearer memory and cognitive difficulties that others start to notice. Scans may show early ventricular enlargement."),
        ("4. ModerateDemented", "Significant memory loss and confusion affecting daily life. Scans often show pronounced atrophy and ventricular enlargement."),
    ]:
        st.markdown(f"**{title}**")
        st.caption(text)
    st.markdown("---")
    st.caption("Simplified educational summary for an academic project — not medical guidance.")


# ============================================================
# Case viewer (used by both tabs; always reads from the DB)
# ============================================================
def decide_and_save(case_id, decision, **kw):
    db.decide(case_id, decision, **kw)
    save_pdf(db.get_case(case_id))  # refresh the patient's cumulative PDF with the final status/edits


def render_case(case_id, prefix):
    c = db.get_case(case_id)
    if not c:
        st.error("Case not found.")
        return

    st.markdown(f"### {c['patient_name']} · `{c['patient_id']}` · `{c['case_id']}`")
    st.caption(f"Scanned {c['created'][:16].replace('T', ' ')} UTC · status: **{c['status']}**"
               + (" (edited)" if c["edited"] else ""))

    col1, col2 = st.columns(2)
    with col1:
        st.image(c["input_image"], caption="Uploaded MRI", width="stretch")
    with col2:
        st.image(c["gradcam_png"], caption="Grad-CAM (model attention)", width="stretch")

    st.subheader(f"Prediction: **{c['stage']}**")
    st.write(f"Confidence: {c['confidence'] * 100:.2f}%")
    st.bar_chart(c["probabilities"])
    st.markdown(f"<p style='font-size:20px; margin-bottom:0;'><strong>Brain region contributing most:</strong> "
                f"{c['region']}</p>", unsafe_allow_html=True)

    st.subheader("🔍 Reason for Detection")
    st.markdown(f"**Reason:** {c['explanation']}")
    st.caption("Rule-based mapping between the Grad-CAM attention zone and typical anatomical patterns per stage. "
               "It describes where the model looked — it is not a diagnosis.")

    st.subheader("📝 Generated report")
    if c["report_source"] == "llm_writer":
        st.caption(f"Written by {OLLAMA_MODEL}, fact-checked by the reviewer "
                   f"({c['review']['attempts']} attempt(s)).")
    else:
        why = "; ".join(c["review"]["issues"]) or "reviewer rejected the draft"
        st.caption(f"Template report (LLM draft not used: {why})")
    for key, title in SECTIONS:
        st.markdown(f"**{title}**")
        st.write(c["report"].get(key, ""))
    st.markdown("**Suggested next steps**")
    for r in c["recommendations"]:
        st.markdown(f"- {r}")
    if c["clinical_note"]:
        st.caption(f"Clinical note provided: {c['clinical_note']}")
    st.caption(DISCLAIMER)

    path = pdf_path(c)
    if not os.path.exists(path) and c["status"] != "rejected":
        save_pdf(c)  # e.g. cases created before the cumulative-PDF change
    st.caption(f"📄 Patient PDF (all scans, newest last) saved automatically: `{path}`")
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button("⬇️ Download patient PDF (all reports)", f.read(), file_name=os.path.basename(path),
                               mime="application/pdf", key=f"{prefix}-pdf-{case_id}")

    # ---- human approval ----
    if c["status"] == "pending":
        st.markdown("#### Review")
        note = st.text_input("Reviewer note (optional)", key=f"{prefix}-note-{case_id}")
        b1, b2 = st.columns(2)
        if b1.button("✅ Approve", key=f"{prefix}-ok-{case_id}", width="stretch"):
            decide_and_save(case_id, "approve", note=note)
            st.rerun()
        if b2.button("🚫 Reject", key=f"{prefix}-no-{case_id}", width="stretch"):
            decide_and_save(case_id, "reject", note=note)
            st.rerun()
        with st.expander("✏️ Edit report, then approve"):
            edited = {k: st.text_area(t, c["report"].get(k, ""), key=f"{prefix}-ed-{k}-{case_id}")
                      for k, t in SECTIONS}
            if st.button("Save edits & approve", key=f"{prefix}-edok-{case_id}"):
                decide_and_save(case_id, "edit", edited_report=edited, note=note)
                st.rerun()
    elif c["status"] == "approved":
        st.success("Approved" + (f" — note: {c['decision_note']}" if c["decision_note"] else ""))
    else:
        st.error("Rejected" + (f" — note: {c['decision_note']}" if c["decision_note"] else ""))


# ============================================================
# Pipeline: image -> prediction -> rules -> report -> SQLite
# ============================================================
def run_pipeline(pid, patient_name, clinical_note, data, mime, pred):
    trend = compute_trend(pred["stage"], db.patient_history(pid), pred["confidence"], db.now_iso())
    with st.spinner(f"Writing and fact-checking the report ({OLLAMA_MODEL} via Ollama)…"):
        report, source, rev = generate_report(patient_name, pred["stage"], pred["confidence"],
                                              pred["probabilities"], pred["region_note"], trend, clinical_note)
    cid = db.save_case({
        "case_id": db.new_case_id(), "patient_id": pid, "created": db.now_iso(), "clinical_note": clinical_note,
        "input_image": data, "input_mime": mime, "gradcam_png": pred["gradcam_png"],
        "stage": pred["stage"], "confidence": pred["confidence"],
        "confidence_band": confidence_band(pred["confidence"]), "probabilities": pred["probabilities"],
        "region": pred["region"], "region_note": pred["region_note"], "explanation": pred["explanation"],
        "trend": trend, "recommendations": recommendations(pred["stage"], trend),
        "report": report, "report_source": source, "review": rev, "status": "pending"})
    save_pdf(db.get_case(cid))  # auto-save PDF (report + confidence + heatmap) into the reports folder
    return cid


# ============================================================
# UI
# ============================================================
st.title("🧠 Alzheimer's Stage Detection & Report Generator")
st.caption("SDP_CISHA_04 — Educational demo. Not for clinical use.")
st.warning("⚠️ Academic project, not a medical device. Not clinically validated; must not be used for real diagnosis.")

tab_new, tab_hist = st.tabs(["🔬 New analysis", "📁 Patients & history"])

with tab_new:
    patients = db.list_patients()
    options = ["➕ New patient"] + [f"{p['name']} ({p['patient_id']})" for p in patients]
    choice = st.selectbox("Patient", options)
    new_name = ""
    if choice == options[0]:
        new_name = st.text_input("New patient name")
    note = st.text_area("Clinical note (optional)", height=70)
    up = st.file_uploader("Upload a brain MRI slice (axial view)", type=["jpg", "jpeg", "png"])
    force = st.checkbox(f"Generate the report even if confidence is below {MIN_CONFIDENCE:.0%}")

    if up is not None:
        st.image(up, caption="Preview", width=200)

    if st.button("Analyze & generate report", type="primary", disabled=up is None):
        if choice == options[0] and not new_name.strip():
            st.error("Enter a name for the new patient.")
        else:
            data = up.getvalue()
            if choice == options[0]:
                pid, pname = None, new_name.strip()
            else:
                p = patients[options.index(choice) - 1]
                pid, pname = p["patient_id"], p["name"]

            image = Image.open(io.BytesIO(data)).convert("RGB")
            with st.spinner("Classifying and building Grad-CAM…"):
                peek = get_classifier().predict(image)
            # confidence gate (original n8n rule): nothing is saved if the model is unsure
            if peek["confidence"] < MIN_CONFIDENCE and not force:
                st.session_state.pop("current_case", None)
                st.warning(f"The model is only {peek['confidence'] * 100:.1f}% confident ({peek['stage']}). "
                           "Upload a clearer axial slice, or tick the box above to proceed with a "
                           "clearly marked low-confidence report.")
            else:
                if pid is None:
                    pid = db.create_patient(pname)
                st.session_state["current_case"] = run_pipeline(pid, pname, note, data, up.type, peek)
                st.rerun()

    if st.session_state.get("current_case"):
        st.markdown("---")
        render_case(st.session_state["current_case"], "new")
    else:
        st.info("Choose a patient, upload an MRI slice and click **Analyze & generate report**.")

with tab_hist:
    patients = db.list_patients()
    labels = {"All patients": None, **{f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in patients}}
    sel = st.selectbox("Filter by patient", list(labels), key="hist-patient")
    cases = db.list_cases(labels[sel])
    if not cases:
        st.info("No saved cases yet.")
    else:
        df = pd.DataFrame(cases)
        df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
        df["created"] = df["created"].str[:16].str.replace("T", " ")
        st.dataframe(df[["created", "case_id", "patient_name", "stage", "confidence", "status", "report_source"]],
                     hide_index=True, width="stretch")
        pick = st.selectbox("Open a case", [c["case_id"] for c in cases], key="hist-case")
        st.markdown("---")
        render_case(pick, "hist")

st.markdown("---")
st.caption("Model: ResNet18 fine-tuned on Kaggle Alzheimer's Dataset · test metrics may be inflated due to known "
           "train/test overlap in the source dataset — see model card for details.")
