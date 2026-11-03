"""Per-patient cumulative PDF (xhtml2pdf, pure Python): every scan's report in one file, oldest first."""
import base64
import html
import io
import os
import re

from report_agent import DISCLAIMER

SECTIONS = [("summary", "Summary"), ("findings", "Imaging findings"), ("interpretation", "Interpretation"),
            ("trend_commentary", "Trend"), ("limitations", "Limitations")]


def _b64(b):
    return base64.b64encode(b).decode()


def _case_html(c, number, total):
    """One scan's report section."""
    e = html.escape
    secs = "".join(f"<h3>{t}</h3><p>{e(c['report'].get(k, ''))}</p>" for k, t in SECTIONS)
    probs = "".join(f"<tr><td>{k}</td><td>{v * 100:.1f}%</td></tr>" for k, v in c["probabilities"].items())
    recs = "".join(f"<li>{e(x)}</li>" for x in c["recommendations"])
    reason = e(c["explanation"].replace("**", ""))
    mime = c.get("input_mime") or "image/png"
    latest = " (latest)" if number == total else ""
    note = f"<p><b>Clinical note:</b> {e(c['clinical_note'])}</p>" if c.get("clinical_note") else ""
    return f"""<h1>Scan {number} of {total}{latest} &ndash; {c['created'][:10]}</h1>
<p>Case: {c['case_id']} | Date: {c['created'][:16].replace('T', ' ')} UTC</p>
<h2>Predicted stage: {c['stage']} ({c['confidence'] * 100:.1f}%)</h2>
<table border="1" cellpadding="3">{probs}</table><br/>
<table><tr>
<td><img src="data:{mime};base64,{_b64(c['input_image'])}" width="220"/><br/>Input MRI</td>
<td><img src="data:image/png;base64,{_b64(c['gradcam_png'])}" width="220"/><br/>Grad-CAM heatmap</td>
</tr></table>
<p><b>Brain region contributing most:</b> {e(c['region'])}</p>
{note}<h3>Reason for detection</h3><p>{reason}</p>
{secs}<h3>Suggested next steps</h3><ul>{recs}</ul>
<p><i>Status: {c['status']}{' (edited by reviewer)' if c['edited'] else ''}.</i></p>"""


def _overview_html(cases):
    """Cover section: scan-by-scan table + the overall direction (computed by rules, not by the LLM)."""
    from explain import CLASSES
    e = html.escape
    rows = "".join(
        f"<tr><td>{i}</td><td>{c['created'][:10]}</td><td>{c['stage']}</td><td>{c['confidence'] * 100:.1f}%</td>"
        f"<td>{e(c['region'] or '')}</td><td>{c['status']}</td></tr>" for i, c in enumerate(cases, 1))
    table = ("<table border='1' cellpadding='3'><tr><th>#</th><th>Date</th><th>Predicted stage</th>"
             f"<th>Confidence</th><th>Main region</th><th>Status</th></tr>{rows}</table>")
    if len(cases) < 2:
        return table
    d = CLASSES.index(cases[-1]["stage"]) - CLASSES.index(cases[0]["stage"])
    word = ("increased (progressed)" if d > 0 else "decreased" if d < 0 else "stayed the same")
    return table + (f"<p><b>Overall:</b> from {cases[0]['stage']} ({cases[0]['created'][:10]}) to "
                    f"{cases[-1]['stage']} ({cases[-1]['created'][:10]}) the model's predicted stage has {word}.</p>")


def patient_report_html(cases):
    """Cumulative report: overview page, then every scan's report, oldest first, newest appended last."""
    e = html.escape
    first = cases[0]
    parts = [f"""<html><body style="font-family:Helvetica;font-size:11pt">
<h1>Alzheimer MRI Screening &ndash; Patient Report</h1>
<p>Patient: {e(first['patient_name'])} ({first['patient_id']}) | Scans on record: {len(cases)}</p>
<h2>Scan history</h2>{_overview_html(cases)}"""]
    for i, c in enumerate(cases, 1):
        parts.append('<pdf:nextpage />' + _case_html(c, i, len(cases)))
    parts.append(f"<hr/><p><i>{e(DISCLAIMER)}</i></p></body></html>")
    return "".join(parts)


def make_pdf(cases) -> bytes:
    from xhtml2pdf import pisa
    buf = io.BytesIO()
    pisa.CreatePDF(patient_report_html(cases), dest=buf)
    return buf.getvalue()


REPORTS_DIR = os.getenv("ALZ_REPORTS_DIR", "reports")


def pdf_path(case):
    """One PDF per patient (named by patient id, so it stays stable); new scans are added to it."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", case["patient_name"]).strip("_") or "patient"
    return os.path.join(REPORTS_DIR, f"{name}_{case['patient_id']}.pdf")


def save_pdf(case) -> str:
    """Rebuild the patient's cumulative PDF from all their non-rejected cases (oldest first) and save it.
    Called after every new scan and after every approve/edit/reject, so the file always matches the database."""
    import db  # local import: avoids a circular import at module load
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = pdf_path(case)
    cases = db.patient_cases(case["patient_id"])
    if not cases:  # every scan of this patient was rejected -> nothing to report
        if os.path.exists(path):
            os.remove(path)
        return path
    with open(path, "wb") as f:
        f.write(make_pdf(cases))
    return path
