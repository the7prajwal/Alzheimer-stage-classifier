# Alzheimer's Stage Detection + Report Generator (single Streamlit site)

One app replaces the old Streamlit demo, the n8n workflows and the FastAPI service.

Flow: upload MRI -> ResNet18 + Grad-CAM -> rules (confidence gate, trend vs previous scans, next steps)
-> Ollama writes the report -> reviewer fact-checks (3 tries, then template) -> saved to SQLite
-> human approve / edit / reject -> PDF download.

Follow-up scans: choose the existing patient and upload the new MRI. The report is written with the patient's earlier scans
as context (direction of change, scan history, low-confidence caveats; a move to a higher stage is reported as increased
severity), and the new report is appended to that patient's single cumulative PDF (overview table + one section per scan,
oldest first). Rejected scans are left out of the PDF.

## Run
    pip install -r requirements.txt
    ollama pull llama3.1:8b      # optional; without Ollama a template report is used
    streamlit run app.py

## Files
- app.py           UI (tabs: New analysis, Patients & history)
- model_utils.py   model download + prediction + Grad-CAM
- explain.py       region zones + "reason" text (no torch)
- report_agent.py  rules, Ollama writer, reviewer, template fallback
- db.py            SQLite storage -> alz_app.db (override with ALZ_DB_PATH)
- pdf_utils.py     one cumulative PDF per patient (<name>_<patient_id>.pdf), auto-saved to ./reports (override with ALZ_REPORTS_DIR)

## Stored per case (table `cases`)
input image (BLOB), Grad-CAM image (BLOB), stage, confidence, all probabilities, peak region, reason text,
trend, recommendations, report sections, report source, reviewer result, status, edit flag, reviewer note, timestamps.
Patients are in table `patients`. Inspect with: `sqlite3 alz_app.db` or DB Browser for SQLite.

Env vars: OLLAMA_URL, OLLAMA_MODEL, ALZ_DB_PATH.
