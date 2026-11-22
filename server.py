"""One command to run everything:   python server.py   ->   http://localhost:8000
Trains all ML models on data/*.csv, serves the interactive dashboard, and answers live prediction requests."""
import json, pathlib, threading, webbrowser, datetime as dt
from flask import Flask, Response, request
from ml_service import Service

HERE = pathlib.Path(__file__).parent
PORT = 8000
app = Flask(__name__)
print("Training models on data/complaints.csv ... (about 10-15 seconds)")
svc = Service()
print(f"Models ready in {svc.train_seconds}s: text classifier, risk model, {len(svc.boots)} bootstrap models for confidence.")
_cache = {"html": None}


def dumps(o):
    return json.dumps(o, separators=(",", ":"), default=float).replace("NaN", "null")


def reply(o, code=200):
    return Response(dumps(o), status=code, mimetype="application/json")


def page():
    if _cache["html"] is None:
        tpl = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
        chart = (HERE / "chart.umd.js").read_text(encoding="utf-8")
        _cache["html"] = tpl.replace("/*__CHARTJS__*/", chart).replace("/*__DATA__*/null", dumps(svc.data()))
    return _cache["html"]


@app.get("/")
def index():
    return Response(page(), mimetype="text/html")


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/api/health")
def health():
    return reply(dict(ok=True, ready=svc.ready, trained_seconds=svc.train_seconds))


@app.get("/api/data")
def data():
    return reply(svc.data())


@app.get("/api/metrics")
def metrics():
    return reply(dict(classifier=svc.clf_perf, risk=svc.risk_perf))


@app.post("/api/classify")
def classify():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text) < 5: return reply(dict(error="Please describe the complaint (at least a few words)."), 400)
    return reply(svc.classify(text))


@app.get("/api/predict")
def predict():
    a = request.args
    try:
        return reply(svc.predict(a["product"], a["category"], float(a.get("dz_moist", 0)), float(a.get("dz_seal", 0))))
    except (KeyError, ValueError) as e:
        return reply(dict(error=f"bad request: {e}"), 400)


@app.post("/api/complaint")
def complaint():
    j = request.get_json(silent=True) or {}
    try:
        for k in ("product", "customer", "text", "category"): assert j.get(k), k
        dt.date.fromisoformat(j.get("date", ""))
    except Exception as e:
        return reply(dict(error=f"missing or invalid field: {e}"), 400)
    cid = svc.add_complaint(dict(date=j["date"], customer=j["customer"], product=j["product"], text=j["text"], category=j["category"], batch_no=j.get("batch_no"), severity=j.get("severity", "Major")))
    _cache["html"] = None
    return reply(dict(ok=True, complaint_id=cid, retrained_seconds=svc.train_seconds))


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"\nOpen the dashboard:  {url}\n(Press Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True)
