"""Build the standalone interactive HTML dashboard (dashboard.html) from the analytics engine.
Run:  python build_html.py     ->  dashboard.html  (works offline, double-click to open)"""
import json, os, pathlib, numpy as np, pandas as pd
import engine as E

CHARTJS = os.environ.get("CHARTJS_PATH", "chart.umd.js")

# ---- playful mission wrapper: emoji, name, how-to steps (the preventive METHOD itself) ----
FUN = {
 "Install / upgrade sieve & magnetic separator": ("🧲", "Magnet Master", ["Inspect the sieve mesh for tears or wear and replace it if in doubt", "Test magnet strength and clean it every shift", "Pass a metal test-piece through the line and log whether it was caught"]),
 "Supplier audit + incoming inspection tightened": ("🕵️", "Supplier Detective", ["Add a visual + lab check to every incoming lot", "Audit the top supplier on this material and share findings", "Reject or quarantine any lot that fails; record why"]),
 "Moisture control at drying + storage SOP": ("💧", "Moisture Patrol", ["Set a maximum moisture limit before storage release", "Measure moisture at the dryer outlet each batch and chart it", "Block release of any lot above the limit until re-dried"]),
 "Fumigation + warehouse pest-control schedule": ("🐛", "Pest Buster", ["Schedule fumigation and trap checks on a fixed calendar", "Inspect stacks, corners and pallets weekly; photograph findings", "Keep a clear gap between stacks and walls"]),
 "Sealing temperature / dwell time standardised": ("🔥", "Seal Sentinel", ["Fix the jaw temperature and dwell time per film type", "Do a seal-strength / leak test every hour and log it", "Stop the line if temperature drifts outside the limit"]),
 "Inline check-weigher with auto-reject": ("⚖️", "Weight Wizard", ["Calibrate filler and check-weigher at start of shift", "Turn on auto-reject for underweight packs", "Review the reject log daily"]),
 "Change-over cleaning checklist + verification": ("🧼", "Clean Sweep", ["Use a signed checklist at every product change-over", "Have a second person verify the line is clear", "Double-check the label against the product before dispatch"]),
 "Process parameter alarm limits added": ("🚨", "Alarm Keeper", ["Define upper and lower limits for critical parameters", "Switch on alarms and make sure someone owns each one", "Review every alarm event weekly to find the pattern"]),
 "Packaging film specification re-qualified": ("📦", "Film Inspector", ["Re-test film thickness, seal strength and barrier on each new lot", "Agree a written spec with the supplier", "Keep a retained sample of every film lot"]),
 "Revalidation of recipe / blend control": ("🧪", "Recipe Guardian", ["Re-verify blend ratios against the approved recipe", "Weigh-and-record each ingredient addition", "Run a uniformity check on the first batch of each shift"]),
 "Preventive maintenance schedule revised": ("🛠️", "Maintenance Hero", ["List parts that wear (jaws, rolls, screws, bolts) and set change intervals", "Inspect for loose or missing metal parts after every service", "Log every service so a repeat can be traced"]),
 "FIFO + dwell-time limits in distribution": ("🚚", "First-In Champion", ["Enforce FIFO in warehouse and dispatch", "Set a maximum dwell time and flag stock nearing it", "Review slow-moving stock monthly"]),
 "Process documentation revised": ("📝", "Doc Doctor", ["Update the SOP to match what really works on the floor", "Walk the team through the change", "Audit one shift for compliance next week"]),
 "Operator retraining only": ("🎓", "Training Time", ["Pair training with a hardware or process fix, not instead of one", "Test understanding on the floor", "Re-check after two weeks"]),
 "Verbal warning to operators": ("🗣️", "Team Talk", ["Back it with a written standard", "Follow up with a process check", "Recognise good performance too"]),
 "Replace customer product only": ("🎁", "Quick Fix", ["Also find and fix the root cause", "Hold the same batch for investigation", "Close only after cause is verified"]),
 "Apology letter to customer": ("✉️", "Customer Care", ["Pair the apology with a concrete preventive action", "Share what changed", "Follow up in a month"])}

def make_data(df=None, mo=None, vec=None, clf=None, cm=None, model=None, rm=None, cur=None, extra=None):
    """Build the dict the dashboard renders. Pass trained objects in (server) or let it train (static build)."""
    if df is None:
        df, mo = E.load(); vec, clf, cm = E.train_classifier(df); df = E.mark_repeats(E.apply_labels(df, vec, clf))
        model, rm, cur, panel = E.train_risk(df, mo); cur["band"] = [E.risk_band(a, b) for a, b in zip(cur.risk_prob, cur.uplift)]
    abr, ac = E.action_by_root_cause(df), E.action_effectiveness(df)
    prods = sorted(df["product"].unique()); custs = sorted(df.customer.unique()); cats = sorted(df.category.unique())
    rcs = sorted(df.root_cause.dropna().unique()); acts = sorted(df.corrective_action.dropna().unique()); sevs = ["Minor", "Major", "Critical"]
    ix = lambda lst: {v: i for i, v in enumerate(lst)}
    ip, ic, ica, irc, iac, isv = ix(prods), ix(custs), ix(cats), ix(rcs), ix(acts), ix(sevs)
    t0 = mo.month.min()
    rec = []
    for r in df.itertuples():
        midx = (r.date.year - t0.year) * 12 + r.date.month - t0.month
        eff = -1 if pd.isna(r.action_effective) else int(bool(r.action_effective))
        rec.append([r.date.year, r.date.month, midx, ip[r.product], ic[r.customer], ica[r.category], irc.get(r.root_cause, -1), iac.get(r.corrective_action, -1),
                    isv[r.severity], eff, int(r.is_repeat), int(r.closure_days) if pd.notna(r.closure_days) else 0])
    line = df.drop_duplicates("product").set_index("product").line.to_dict()
    # ---- missions (the gamified check-in): top risks -> proven preventive actions
    missions, seen = [], set()
    for _, rr in cur.sort_values("risk_prob", ascending=False).iterrows():
        if rr.band == "Low" and len(missions) >= 3: continue
        recs, gen = E.recommend(df, abr, rr["product"], rr.category, top_rc=3)
        recs = recs[recs.recommendation == "Prioritise"].drop_duplicates("action")
        n_cell = 0
        for _, a in recs.iterrows():
            key = (rr["product"], a.action)
            if key in seen or n_cell >= 2 or len(missions) >= 9: continue
            seen.add(key); n_cell += 1; emoji, name, steps = FUN[a.action]
            missions.append(dict(id=f"m{len(missions)+1}", product=rr["product"], category=rr.category, risk=round(float(rr.risk_prob), 3), uplift=float(rr.uplift),
                                 root_cause=a.likely_root_cause, likelihood=float(a.likelihood_pct), action=a.action, eff=float(a.historical_effectiveness_pct),
                                 emoji=emoji, name=name, steps=steps, line=line[rr["product"]]))
        if len(missions) >= 9: break
    kp = E.kpis(df, mo); an = E.annual_rate(df, mo)
    ch = E.chronic_table(df).head(15)
    ew = E.early_warnings(cur); pr = E.process_risk(df, cur)
    data = dict(
        asof=str(df.date.max().date()), forecast=cur.forecast_month.iat[0], n=len(df), t0=[t0.year, t0.month],
        prods=prods, custs=custs, cats=cats, rcs=rcs, acts=acts, sevs=sevs, line=[line[p] for p in prods], rec=rec,
        risk=[dict(p=ip[r["product"]], c=ica[r.category], prob=round(float(r.risk_prob), 3), base=round(float(r.baseline_prob), 3), up=float(r.uplift), band=r.band) for _, r in cur.iterrows()],
        proc=pr.head(14).round(3).to_dict("records"), area=E.area_risk(pr).round(3).to_dict("records"),
        lines=E.line_risk(df, cur).round(3).to_dict("records"), ew=ew.to_dict("records"),
        b_inf=E.parameter_bands(df, mo, "Infestation", "avg_moisture").round(3).to_dict("records"),
        b_pack=E.parameter_bands(df, mo, "Packaging Issue", "avg_seal_temp").round(3).to_dict("records"),
        chronic=ch.round(1).to_dict("records"), actions=ac.round(1).to_dict("records"), themes=E.themes(df).to_dict("records"),
        kpi={k: (float(v) if not isinstance(v, (int, float)) else v) for k, v in kp.items()},
        annual=an.round(4).to_dict("records"), missions=missions, fun={k: dict(e=v[0], n=v[1], s=v[2]) for k, v in FUN.items()},
        generic=E.GENERIC,
        metrics=dict(acc=round(cm["accuracy"], 4), f1=round(cm["macro_f1"], 4), n_test_c=cm["n_test"], auc=round(rm["auc"], 3), brier=round(rm["brier"], 4),
                     brier_naive=round(rm["brier_naive"], 4), test_from=rm["test_from"], n_test_r=rm["n_test"],
                     unlabeled=round(float((df.category_source == "AI-classified").mean()), 3)))
    if extra: data.update(extra)
    data["_missions"] = len(missions)
    return data

def build():
    from ml_service import Service      # static file also carries real metrics and confidence intervals
    data = Service().data(); data["live"] = False
    tpl = pathlib.Path("dashboard_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__CHARTJS__*/", pathlib.Path(CHARTJS).read_text(encoding="utf-8")).replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=float))
    pathlib.Path("dashboard.html").write_text(html, encoding="utf-8")
    print("dashboard.html", round(len(html) / 1024), "KB;", data["_missions"], "missions;", len(data["rec"]), "records")

if __name__ == "__main__":
    build()
