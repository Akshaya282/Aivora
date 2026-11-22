# Aivora · Zero Customer Complaint System

**AI-based complaint analytics, root-cause identification, risk prediction and prevention**
Team Aivora · Team ID SKH2-2026-01018 · Sakthi Hackathon 2.0, Round 2 · Problem: ABT-CPD Zero Customer Complaint

> From 19 years of complaints to complaints that never happen.

---

## 1. What it does

A working prototype that turns a complaint history into **insight, prediction and prevention**, and covers all nine requirements of the problem statement:

| # | Requirement | Where to see it in the dashboard |
|---|---|---|
| 1 | Classify complaints into meaningful categories | **Live AI** (text classifier with confidence), **Pareto & Themes** (clustering) |
| 2 | Pareto / major contributors | **Pareto & Themes** (by category, product, customer, root cause, line) |
| 3 | Recurring complaints and chronic problem areas | **Chronic & Actions** |
| 4 | Product / customer / complaint type / root-cause relationships | **Relationships** (heatmaps, drill-down, lift tables, process-parameter bands) |
| 5 | High-risk products and processes | **Risk & Early Warning** (product heatmap, process failure modes, process areas, production lines) |
| 6 | Predict complaint categories and risk | **Risk & Early Warning** (next-month probability with 90% interval and confidence) |
| 7 | Recommend preventive / corrective actions | **Preventive Advisor**, **Prevention Quest** |
| 8 | Management dashboard | The whole app (trends, alerts, KPIs) |
| 9 | Measurable KPIs towards zero | **Overview & KPIs** (scorecard and glide-path) |

Extra: a **Model Trust** tab (sensitivity, specificity, calibration), **what-if sliders**, and the game-style **Prevention Quest** that asks the plant team whether each proven fix is in place (Done / Working on it / Not yet) with XP, levels and badges.

---

## 2. Quick start (Windows, Mac or Linux)

You need **Python 3.10 or newer** (python.org; on Windows tick "Add Python to PATH").

1. Unzip the project and open a terminal **inside the `zcc` folder**.
2. Run:
   ```
   pip install -r requirements.txt
   python server.py
   ```
   Or on Windows double-click **`run.bat`**; on Mac/Linux run `./run.sh`.
3. Wait about 15 seconds. Your browser opens **http://localhost:8000**. Keep the terminal open; press Ctrl+C to stop.

Opening `dashboard.html` directly also works, but it is a snapshot: the live features (Live AI, what-if sliders, save-and-retrain) need `python server.py`.

---

## 3. Using public data (openFDA food recalls)

The bundled demo data is **synthetic**. To run on a public dataset instead:

1. Download the Food Enforcement dataset from https://open.fda.gov/apis/food/enforcement/download/
2. Put the file in the `zcc` folder, then run:
   ```
   python import_fda.py
   python server.py --data data_fda
   ```
   Or on Windows double-click **`run_fda.bat`** (Mac/Linux: `./run_fda.sh`).

What the importer does, and what it cannot know:

- Date = recall initiation date; severity = FDA class (I = Critical, II = Major, III = Minor); customer = recalling firm.
- Product group, complaint category and root cause are **derived from the recall text by keyword rules** (`fda_rules.py`), because the FDA does not publish them.
- Shipment volume, process parameters and corrective-action outcomes do **not** exist in public data. Rates are per product-month, process charts are skipped, and prevention guidance is curated good practice, not learned effectiveness.
- The classifier score on this data measures agreement with those same rules, so it is **not** an independent accuracy figure.

---

## 4. Project structure

```
zcc/
├── server.py               Flask app: dashboard + live prediction API (start here)
├── ml_service.py           Trains all models, specificity/confidence metrics, predict, classify, retrain
├── engine.py               Core analytics: classifier, clustering, Pareto, chronic score, risk model, KPIs
├── build_html.py           Builds the data the dashboard renders (and the static dashboard.html)
├── dashboard_template.html The interactive dashboard (Chart.js)
├── dashboard.html          Offline snapshot of the dashboard
├── app.py                  Older Streamlit dashboard (optional)
├── generate_data.py        Creates the synthetic 19-year dataset in data/
├── import_fda.py           Converts an openFDA JSON/ZIP into data_fda/
├── fda_rules.py            Keyword rules + curated preventive guidance for public data
├── data/                   complaints.csv, monthly.csv
└── requirements.txt, run.bat, run.sh, run_fda.bat, run_fda.sh
```

---

## 5. The ML pipeline

### Preprocessing
- **Dates:** parsed and bucketed to months; seasonality features.
- **Missing values:** records with no category are filled by the text classifier (flagged AI-classified); records with no root cause or action stay in the counts but are left out of cause and action analytics.
- **Text:** TF-IDF on 1- and 2-word phrases (sublinear term frequency, rare terms dropped).
- **Repeat flag:** same product, category and root cause within the previous 12 months.
- **Risk features** (per product × category × month): 1-, 3- and 12-month counts, 6-month average, seasonality, moisture, sealing temperature, shipment volume, rolling 36-month z-scores of process parameters.
- **Target and split:** "at least one complaint next month"; train/test split **by time** (last 36 months held out).

### Models
| Model | Method | Purpose |
|---|---|---|
| Complaint classifier | TF-IDF + Logistic Regression | Category and confidence for free-text complaints |
| Theme discovery | TF-IDF + K-Means (9 clusters) | Recurring complaint wording without labels |
| Risk predictor | Gradient boosting | Next-month probability per product × category |
| Confidence | 15 bootstrap risk models | 90% interval, High/Medium/Low label, model agreement |
| Chronic score, action effectiveness, process risk | Statistics + rules | Persistent problems, which actions work, which processes drive risk |

### Measured on held-out data (synthetic demo dataset)
| Model | Metric | Value |
|---|---|---|
| Classifier | Accuracy / macro sensitivity / macro specificity | 97.4% / 98.0% / 99.7% |
| Risk model | AUC (90% CI) | 0.69 (0.67 to 0.71) |
| Risk model | Sensitivity / specificity at the alert threshold | 56% / 71% |
| Risk model | Precision / negative predictive value | 45% / 79% |
| Risk model | Calibration | Good: a 70% prediction comes true about 70% of the time |

The alert threshold is chosen on a validation year to cap false alarms at 30%, never on the test period.

---

## 6. Using your own (real ABT-CPD) data

Replace `data/complaints.csv` and `data/monthly.csv`, then restart `server.py`.

**complaints.csv:** `complaint_id, date, customer, product, line, batch_no, complaint_text, category, severity (Minor/Major/Critical), batch_moisture, batch_seal_temp, root_cause, corrective_action, action_effective, closure_days`. Only `date, customer, product, line, complaint_text, severity` are essential; `category` can be blank (the classifier fills it).

**monthly.csv:** `product, line, month, volume_tonnes, avg_moisture, avg_seal_temp` for every product and month.

New complaints can also be added in the app (**Live AI → Save and retrain**).

---

## 7. API (http://localhost:8000)

| Endpoint | Purpose |
|---|---|
| `GET /` | The dashboard |
| `GET /api/data` | Full dashboard data |
| `GET /api/metrics` | Classifier and risk-model metrics |
| `POST /api/classify` | Category, confidence, top-3 alternatives |
| `GET /api/predict?product=..&category=..` | Next-month probability, interval, confidence, what-if |
| `POST /api/complaint` | Save a complaint and retrain |

---

## 8. Honest limitations

- The bundled data is **synthetic**; re-validate on real ABT-CPD records.
- The risk model is moderate (AUC 0.69): use it to prioritise inspections, not as a verdict. Low-confidence alerts mean the models disagree.
- What-if sliders move predictions only modestly, because recent complaint history dominates next-month risk.
- The Quest's "risk shield" is an estimate assuming each fix delivers half its historical effectiveness.
- Indian public data (FSSAI / data.gov.in) is yearly totals, not complaint-level records, so it cannot train these models.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `python is not recognized` | Reinstall Python and tick "Add Python to PATH", or use `py server.py` |
| Port 8000 already in use | Close any other running copy, then restart |
| Live AI / what-if disabled | You opened `dashboard.html` directly; use `python server.py` |
| `import_fda.py` finds no file | Put the `.json` or `.zip` in the `zcc` folder or pass its path |

---

Built with Python, pandas, scikit-learn, Flask and Chart.js. Developed with AI assistance; models and numbers were validated by the team.

*Team Aivora · Sakthi Hackathon 2.0 · MCET Student Research Council – Innovation Nexus*
Team Aivora · Sakthi Hackathon 2.0 · MCET Student Research Council – Innovation Nexus
