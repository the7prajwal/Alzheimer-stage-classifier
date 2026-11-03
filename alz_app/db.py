"""Local SQLite storage: patients + cases (input image, heatmap, prediction, reason, report, approval)."""
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.getenv("ALZ_DB_PATH", "alz_app.db")

SCHEMA = """
create table if not exists patients(
    patient_id text primary key,
    name       text not null,
    created    text not null
);
create table if not exists cases(
    case_id          text primary key,
    patient_id       text not null references patients(patient_id),
    created          text not null,
    clinical_note    text default '',
    input_image      blob not null,
    input_mime       text,
    gradcam_png      blob not null,
    stage            text not null,
    confidence       real not null,
    confidence_band  text,
    probabilities    text not null,   -- json
    region           text,
    region_note      text,
    explanation      text,
    trend            text,            -- json
    recommendations  text,            -- json list
    report           text,            -- json {summary, findings, ...}
    report_source    text,            -- llm_writer | template_fallback
    review           text,            -- json {passed, attempts, issues}
    status           text not null default 'pending',   -- pending | approved | rejected
    edited           integer default 0,
    decision_note    text default '',
    decided          text
);
create index if not exists idx_cases_patient on cases(patient_id, created);
"""

_JSON_FIELDS = ("probabilities", "trend", "recommendations", "report", "review")


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("pragma foreign_keys=on")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    with _conn() as c:
        c.executescript(SCHEMA)


# ---------- patients ----------
def create_patient(name: str) -> str:
    pid = "ALZ-" + uuid.uuid4().hex[:6].upper()
    with _conn() as c:
        c.execute("insert into patients values(?,?,?)", (pid, name.strip(), now_iso()))
    return pid


def list_patients():
    with _conn() as c:
        return [dict(r) for r in c.execute("select patient_id, name from patients order by created desc")]


def get_patient(pid):
    with _conn() as c:
        r = c.execute("select patient_id, name from patients where patient_id=?", (pid,)).fetchone()
    return dict(r) if r else None


# ---------- cases ----------
def patient_history(pid, n=10):
    """Previous non-rejected scans, newest first (used for the trend)."""
    with _conn() as c:
        rows = c.execute(
            "select case_id, created, stage, confidence, region from cases "
            "where patient_id=? and status!='rejected' order by created desc, rowid desc limit ?",
            (pid, n)).fetchall()
    return [{"case_id": r["case_id"], "created": r["created"], "date": r["created"][:10], "stage": r["stage"],
             "confidence": r["confidence"], "region": r["region"]} for r in rows]


def patient_cases(pid):
    """All non-rejected cases of a patient (full rows incl. images), oldest first. Feeds the cumulative PDF."""
    with _conn() as c:
        ids = [r["case_id"] for r in c.execute(
            "select case_id from cases where patient_id=? and status!='rejected' order by created, rowid", (pid,))]
    return [get_case(i) for i in ids]


def new_case_id():
    return f"CASE-{int(time.time())}-{uuid.uuid4().hex[:4].upper()}"


def save_case(d: dict) -> str:
    row = dict(d)
    for k in _JSON_FIELDS:
        row[k] = json.dumps(row.get(k))
    cols = ["case_id", "patient_id", "created", "clinical_note", "input_image", "input_mime", "gradcam_png",
            "stage", "confidence", "confidence_band", "probabilities", "region", "region_note", "explanation",
            "trend", "recommendations", "report", "report_source", "review", "status"]
    with _conn() as c:
        c.execute(f"insert into cases({','.join(cols)}) values({','.join('?' * len(cols))})",
                  [row.get(k) for k in cols])
    return d["case_id"]


def _decode(r):
    d = dict(r)
    for k in _JSON_FIELDS:
        d[k] = json.loads(d[k]) if d.get(k) else None
    d["edited"] = bool(d["edited"])
    return d


def get_case(cid):
    with _conn() as c:
        r = c.execute("select cases.*, patients.name as patient_name from cases "
                      "join patients using(patient_id) where case_id=?", (cid,)).fetchone()
    return _decode(r) if r else None


def list_cases(pid=None):
    """Light listing (no image blobs)."""
    q = ("select case_id, patient_id, patients.name as patient_name, cases.created as created, stage, confidence, "
         "status, report_source from cases join patients using(patient_id)")
    args = ()
    if pid:
        q += " where patient_id=?"
        args = (pid,)
    with _conn() as c:
        return [dict(r) for r in c.execute(q + " order by cases.created desc", args)]


def decide(cid, decision, edited_report=None, note=""):
    assert decision in ("approve", "edit", "reject")
    case = get_case(cid)
    report = case["report"]
    if decision == "edit" and edited_report:
        report = {**report, **edited_report}
    status = "rejected" if decision == "reject" else "approved"
    with _conn() as c:
        c.execute("update cases set report=?, status=?, edited=?, decision_note=?, decided=? where case_id=?",
                  (json.dumps(report), status, int(decision == "edit"), note, now_iso(), cid))
