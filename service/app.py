"""Classifier + history + PDF service. n8n calls this; it does NOT write the report text."""
import base64, html, io, json, os, sqlite3, time, uuid
from datetime import datetime, timezone
import numpy as np, torch, torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms

REPO = "the7prajwal/alzheimer-stage-resnet18"
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DB = os.getenv("DB_PATH", "alz_reports.db")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(hf_hub_download(REPO, "pytorch_model.bin"), map_location=device))
model.to(device).eval()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
with db() as c:
    c.executescript("""create table if not exists patients(patient_id text primary key, name text, created real);
    create table if not exists cases(case_id text primary key, patient_id text, created real, status text, data text);""")

app = FastAPI(title="Alzheimer report service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def peak_region(g):  # same zones as the Streamlit app
    h, w = g.shape; cy, cx = h / 2, w / 2
    py, px = np.unravel_index(np.argmax(g), g.shape)
    r = np.sqrt((py - cy) ** 2 + (px - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
    if r <= 0.35: return "ventricular / central", "the central region near the ventricles"
    if r <= 0.7: return "hippocampal / temporal", "the medial temporal area associated with the hippocampus"
    return "cortical", "the outer cortical surface"

def history(pid, n=5):
    with db() as c:
        rows = c.execute("select data from cases where patient_id=? and status!='rejected' order by created desc limit ?", (pid, n)).fetchall()
    out = []
    for r in rows:
        d = json.loads(r["data"])
        out.append({"case_id": d["case_id"], "date": d["created"][:10], "stage": d["prediction"]["stage"], "confidence": d["prediction"]["confidence"]})
    return out

class PredictIn(BaseModel):
    patient_name: str = ""
    patient_id: str | None = None
    image_base64: str

@app.post("/predict")
def predict(b: PredictIn):
    try: img = Image.open(io.BytesIO(base64.b64decode(b.image_base64.split(",")[-1]))).convert("RGB")
    except Exception: raise HTTPException(422, "Invalid image")
    with db() as c:
        pid = b.patient_id
        if pid:
            row = c.execute("select name from patients where patient_id=?", (pid,)).fetchone()
            if not row: raise HTTPException(404, "Unknown patient_id")
            name = row["name"]
        else:
            if not b.patient_name.strip(): raise HTTPException(422, "patient_name required for a new patient")
            pid, name = "ALZ-" + uuid.uuid4().hex[:6].upper(), b.patient_name.strip()
            c.execute("insert into patients values(?,?,?)", (pid, name, time.time()))
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad(): probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    i = int(probs.argmax())
    g = cam(input_tensor=x, targets=None)[0]
    rgb = np.clip(x[0].cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN), 0, 1)
    buf = io.BytesIO(); Image.fromarray(show_cam_on_image(rgb, g, use_rgb=True)).save(buf, "PNG")
    region, note = peak_region(g)
    return {"patient_id": pid, "patient_name": name,
            "prediction": {"stage": CLASSES[i], "confidence": float(probs[i]), "probabilities": {k: float(p) for k, p in zip(CLASSES, probs)}},
            "region": region, "region_note": note,
            "gradcam_png_base64": base64.b64encode(buf.getvalue()).decode(), "history": history(pid)}

def get_case(cid):
    with db() as c: r = c.execute("select data from cases where case_id=?", (cid,)).fetchone()
    if not r: raise HTTPException(404, "Case not found")
    return json.loads(r["data"])

def put_case(d):
    with db() as c:
        c.execute("insert or replace into cases values(?,?,?,?,?)",
                  (d["case_id"], d["patient"]["id"], time.time(), d["approval"]["status"], json.dumps(d)))

@app.post("/cases")
def save_case(case: dict):
    put_case(case); return case

class Decision(BaseModel):
    decision: str
    edited_report: dict | None = None
    note: str = ""

@app.post("/cases/{cid}/decision")
def decide(cid: str, b: Decision):
    d = get_case(cid)
    if b.decision not in ("approve", "edit", "reject"): raise HTTPException(422, "decision must be approve|edit|reject")
    if b.decision == "edit" and b.edited_report: d["report"].update(b.edited_report)
    status = "rejected" if b.decision == "reject" else "approved"
    d["approval"] = {"status": status, "edited": b.decision == "edit", "note": b.note,
                     "decided": datetime.now(timezone.utc).isoformat(),
                     "pdf_url": f"/reports/{cid}/pdf" if status == "approved" else None}
    put_case(d); return d

@app.get("/patients")
def patients():
    with db() as c: return [dict(r) for r in c.execute("select patient_id, name from patients order by created desc")]

@app.get("/patients/{pid}/history")
def patient_history(pid: str): return history(pid, 50)

def report_html(c):
    e, r, p = html.escape, c["report"], c["prediction"]
    secs = "".join(f"<h3>{t}</h3><p>{e(r.get(k, ''))}</p>" for k, t in [("summary", "Summary"), ("findings", "Imaging findings"),
             ("interpretation", "Interpretation"), ("trend_commentary", "Trend"), ("limitations", "Limitations")])
    probs = "".join(f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>" for k, v in p["probabilities"].items())
    recs = "".join(f"<li>{e(x)}</li>" for x in c["recommendations"])
    return f"""<html><body style="font-family:Helvetica;font-size:11pt">
<h1>Alzheimer MRI Screening Report</h1>
<p>Patient: {e(c['patient']['name'])} ({c['patient']['id']}) | Case: {c['case_id']} | Date: {c['created'][:10]}</p>
<h2>Predicted stage: {p['stage']} ({p['confidence']*100:.1f}%)</h2>
<table border="1" cellpadding="3">{probs}</table><br/>
<img src="data:image/png;base64,{c['gradcam_png_base64']}" width="240"/>
{secs}<h3>Suggested next steps</h3><ul>{recs}</ul><p><i>{e(c['disclaimer'])}</i></p></body></html>"""

@app.get("/reports/{cid}/pdf")
def pdf(cid: str):
    c = get_case(cid)
    if c["approval"]["status"] != "approved": raise HTTPException(409, "Report not approved")
    from xhtml2pdf import pisa
    buf = io.BytesIO(); pisa.CreatePDF(report_html(c), dest=buf)
    return Response(buf.getvalue(), media_type="application/pdf")
