"""Port of the n8n flow: rules decide the facts, Ollama only phrases, a reviewer fact-checks (max 3 tries),
then a template fallback. Works without Ollama too (falls back to the template)."""
import json
import os
import re
from datetime import datetime

import requests

from explain import CLASSES as STAGES, confidence_band

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MIN_CONFIDENCE = 0.55
KEYS = ["summary", "findings", "interpretation", "trend_commentary", "limitations"]

NEXT = {
    "NonDemented": ["No imaging pattern suggestive of dementia was flagged by the model.",
                    "Continue routine check-ups as advised by a clinician."],
    "VeryMildDemented": ["Subtle early changes were flagged; discuss with a neurologist or memory clinic.",
                         "Clinician-administered cognitive screening is commonly used to follow up."],
    "MildDemented": ["Consult a neurologist for clinical evaluation and cognitive testing.",
                     "Discuss support planning and home safety with the care team."],
    "ModerateDemented": ["Prompt specialist evaluation is advised.",
                         "Involve family or caregivers in care planning and daily-living support."],
}
FIND = {
    "NonDemented": "no notable atrophy pattern",
    "VeryMildDemented": "subtle, early structural changes",
    "MildDemented": "moderate structural changes, including some ventricular enlargement",
    "ModerateDemented": "pronounced atrophy with ventricular enlargement",
}
DISCLAIMER = ("Educational AI-generated screening report. Not a medical device and not a diagnosis; "
              "must be reviewed by a qualified clinician.")


# ---------- rules (code decides) ----------
def _days_between(a_iso, b_iso):
    try:
        a = datetime.fromisoformat(a_iso)
        b = datetime.fromisoformat(b_iso)
        return max((b - a).days, 0)
    except Exception:
        return None


def compute_trend(stage, history, confidence=None, now_iso=None):
    """history = this patient's earlier non-rejected scans, newest first (db.patient_history).
    Code decides direction and wording of the change; the LLM only phrases it."""
    if not history:
        return {"has_previous": False}
    chrono = list(reversed(history))                      # oldest -> newest
    prev, first = history[0], chrono[0]
    d = STAGES.index(stage) - STAGES.index(prev["stage"])
    change = "progressed" if d > 0 else "improved" if d < 0 else "unchanged"
    net = STAGES.index(stage) - STAGES.index(first["stage"])
    seq = [STAGES.index(h["stage"]) for h in chrono] + [STAGES.index(stage)]
    steps = [b - a for a, b in zip(seq, seq[1:])]
    if all(x == 0 for x in steps):
        overall = "stable"
    elif net > 0 and all(x >= 0 for x in steps):
        overall = "progressing"
    elif net < 0 and all(x <= 0 for x in steps):
        overall = "lower_predicted_stage"
    else:
        overall = "fluctuating"

    if change == "progressed":
        statement = (f"The predicted stage moved from {prev['stage']} ({prev['date']}) to {stage}, "
                     f"{abs(d)} stage(s) higher: the model's predicted disease severity has INCREASED "
                     f"(progressed) since the previous scan.")
    elif change == "improved":
        statement = (f"The predicted stage moved from {prev['stage']} ({prev['date']}) to {stage}, "
                     f"{abs(d)} stage(s) lower. Alzheimer's is normally progressive, so this more likely reflects "
                     f"differences between the scans (slice, image quality) than true recovery; treat it cautiously.")
    else:
        statement = f"The predicted stage is unchanged ({stage}) compared with the previous scan on {prev['date']}."

    caveat = ""
    if confidence is not None and "low" in (confidence_band(confidence), confidence_band(prev["confidence"])):
        caveat = "At least one of the two scans has low model confidence, so the apparent change is tentative."

    return {
        "has_previous": True, "previous_stage": prev["stage"], "previous_date": prev["date"],
        "previous_confidence": prev["confidence"], "change": change, "stages_changed": abs(d),
        "direction_statement": statement, "caveat": caveat,
        "days_since_previous": _days_between(prev.get("created", ""), now_iso) if now_iso else None,
        "first_scan_stage": first["stage"], "first_scan_date": first["date"],
        "net_change_since_first": "progressed" if net > 0 else "lower" if net < 0 else "same",
        "overall_pattern": overall, "scans_on_record": len(history) + 1,
        "history": [{"date": h["date"], "stage": h["stage"], "confidence_pct": round(h["confidence"] * 100, 1),
                     "region": h.get("region")} for h in chrono],
    }


def recommendations(stage, trend=None):
    recs = list(NEXT[stage])
    if trend and trend.get("has_previous"):
        if trend["change"] == "progressed":
            recs.append("The predicted stage has advanced since the previous scan; earlier review by a "
                        "neurologist is advisable.")
        elif trend["change"] == "improved":
            recs.append("The predicted stage is lower than before; a clinician should compare the scans directly "
                        "before drawing any conclusion.")
        else:
            recs.append("The predicted stage is unchanged; continue the follow-up schedule set by the care team.")
    return recs + ["This is a model screening output; a qualified clinician must confirm any finding."]


# ---------- writer ----------
def _messages(facts, feedback):
    sys = ('You write a cautious AI-screening report for an academic brain-MRI classifier. Use ONLY the facts JSON. '
           'Never invent numbers, stages, measurements or history. Never state a diagnosis; use wording like '
           '"the model predicts" and "consistent with". Return a JSON object with string keys: summary, findings, '
           'interpretation, trend_commentary, limitations (2-3 sentences each). If trend.has_previous is false, '
           'trend_commentary must say no earlier scan is on record. If confidence_band is low, say the result '
           'should be treated cautiously. '
           'If trend.has_previous is true, this is a FOLLOW-UP report: take the earlier scans in trend.history into '
           'account in summary, interpretation and trend_commentary. trend_commentary must name the previous stage '
           'and date, restate trend.direction_statement faithfully (a move to a higher stage means the predicted '
           'disease severity increased/progressed; a move to a lower stage must NOT be called recovery), mention '
           'the overall_pattern across the scans on record, and repeat trend.caveat if it is not empty. '
           'Do not mention any scan, date or stage that is not in the facts.')
    user = "FACTS:\n" + json.dumps(facts)
    if feedback:
        user += ("\n\nYour previous draft was rejected by the reviewer: " + "; ".join(feedback) +
                 ". Fix every issue and return the full JSON again.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _call_ollama(messages):
    r = requests.post(OLLAMA_URL, timeout=300, json={
        "model": OLLAMA_MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0.2}, "messages": messages})
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


# ---------- reviewer (code checks) ----------
def review(raw, stage, probabilities, confidence, trend):
    issues, draft = [], None
    try:
        draft = json.loads(raw)
        if not isinstance(draft, dict):
            raise ValueError
    except Exception:
        return None, ["Output was not valid JSON"]
    for k in KEYS:
        if not isinstance(draft.get(k), str) or len(draft[k].strip()) < 20:
            issues.append(f"Missing or too short section: {k}")
    text = " ".join(str(draft.get(k, "")) for k in KEYS)
    allowed = [confidence * 100, *[v * 100 for v in probabilities.values()]]
    if trend.get("has_previous"):
        allowed += [trend["previous_confidence"] * 100, *[h["confidence_pct"] for h in trend["history"]]]
    for m in re.findall(r"\d+(?:\.\d+)?\s*%", text):
        n = float(m.replace("%", "").strip())
        if not any(abs(a - n) <= 0.6 for a in allowed):
            issues.append(f"Percentage {m} is not in the supplied facts")
    if stage.lower() not in re.sub(r"\s", "", str(draft.get("summary", "")).lower()):
        issues.append(f"Summary must name the predicted stage {stage}")
    if re.search(r"(diagnos(ed|is) with|has alzheimer|you have|confirmed (alzheimer|dementia))", text, re.I):
        issues.append('Contains diagnostic language; use "model predicts / consistent with"')
    tc = str(draft.get("trend_commentary", ""))
    if (not trend["has_previous"] and re.search(r"(previous|earlier|prior) scan", tc, re.I)
            and not re.search(r"no (earlier|previous|prior)", tc, re.I)):
        issues.append("Mentions a previous scan but none is on record")
    if trend.get("has_previous"):
        norm = re.sub(r"[^a-z]", "", tc.lower())
        if trend["previous_stage"].lower() not in norm:
            issues.append(f"trend_commentary must name the previous stage {trend['previous_stage']}")
        ch = trend["change"]
        if ch == "progressed":
            if not re.search(r"progress|worsen|advanc|increas|higher", tc, re.I):
                issues.append("trend_commentary must say the predicted severity increased (progressed)")
            if re.search(r"\b(improved|improvement|recover(ed|y)|better)\b", tc, re.I):
                issues.append("trend_commentary calls a progression an improvement")
        elif ch == "improved":
            if re.search(r"\b(progressed|worsened|advanced|increased)\b", tc, re.I):
                issues.append("trend_commentary says progression but the predicted stage is lower")
            if re.search(r"\b(recovered|recovery|cured|reversed)\b", tc, re.I):
                issues.append("trend_commentary implies recovery; say it is likely a scan difference")
        else:
            if not re.search(r"unchanged|same|stable|no change|remain", tc, re.I):
                issues.append("trend_commentary must say the predicted stage is unchanged")
            if re.search(r"\b(progressed|worsened|advanced|improved)\b", tc, re.I):
                issues.append("trend_commentary reports a change but the predicted stage is unchanged")
    return draft, issues


def template_report(stage, confidence, band, region_note, trend):
    pct = f"{confidence * 100:.1f}"
    summary = f"The model classified this scan as {stage} with {band} confidence ({pct}%)."
    interp = f"This pattern is consistent with {FIND[stage]}. It is a screening output, not a diagnosis."
    if trend["has_previous"]:
        trail = " -> ".join(f"{h['stage']} ({h['date']})" for h in trend["history"])
        tc = (f"{trend['direction_statement']} Scans on record: {trail} -> {stage} (this scan). "
              f"Overall pattern: {trend['overall_pattern'].replace('_', ' ')}. {trend['caveat']}").strip()
        if trend["change"] == "progressed":
            summary += " Compared with the previous scan, the predicted stage has progressed."
            interp += " Taken together with the earlier scans, this suggests increasing disease severity."
        elif trend["change"] == "unchanged":
            summary += " The predicted stage is unchanged since the previous scan."
    else:
        tc = "No earlier scan is on record for this patient."
    return {
        "summary": summary,
        "findings": f"Model attention was concentrated in {region_note}.",
        "interpretation": interp,
        "trend_commentary": tc,
        "limitations": "Academic model, not clinically validated; the training dataset has known train/test overlap.",
    }


def generate_report(patient_name, stage, confidence, probabilities, region_note, trend, clinical_note="",
                    llm=_call_ollama):
    """Returns (report_dict, source, review_info). `llm` is injectable for tests."""
    band = confidence_band(confidence)
    facts = {"patient_name": patient_name, "stage": stage, "confidence_pct": round(confidence * 100, 1),
             "probabilities_pct": {k: round(v * 100, 1) for k, v in probabilities.items()},
             "confidence_band": band, "gradcam_region": region_note, "trend": trend,
             "clinical_note": clinical_note}
    fallback = template_report(stage, confidence, band, region_note, trend)
    feedback, issues = [], []
    for attempt in range(1, 4):
        try:
            raw = llm(_messages(facts, feedback))
        except Exception as e:  # Ollama not running, timeout, etc.
            return fallback, "template_fallback", {"passed": False, "attempts": attempt,
                                                   "issues": [f"Ollama unavailable: {e}"]}
        draft, issues = review(raw, stage, probabilities, confidence, trend)
        if not issues:
            return {k: draft[k].strip() for k in KEYS}, "llm_writer", {"passed": True, "attempts": attempt, "issues": []}
        feedback = issues
    return fallback, "template_fallback", {"passed": False, "attempts": 3, "issues": issues}
