"""Analytics + ML engine for the Zero Customer Complaint system (ABT-CPD prototype)."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = "data/complaints.csv"; MONTHLY = "data/monthly.csv"

# ---------- load ----------
def load():
    df = pd.read_csv(DATA, parse_dates=["date"]); mo = pd.read_csv(MONTHLY, parse_dates=["month"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df, mo

# ---------- 1. NLP classification ----------
def train_classifier(df):
    lab = df[df.category.notna()]
    X_tr, X_te, y_tr, y_te = train_test_split(lab.complaint_text, lab.category, test_size=.2, random_state=1, stratify=lab.category)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=8)
    Xtr = vec.fit_transform(X_tr); clf.fit(Xtr, y_tr)
    pred = clf.predict(vec.transform(X_te))
    metrics = dict(accuracy=accuracy_score(y_te, pred), macro_f1=f1_score(y_te, pred, average="macro"), n_train=len(X_tr), n_test=len(X_te))
    vec.fit(lab.complaint_text); clf.fit(vec.transform(lab.complaint_text), lab.category)  # final fit on all labelled
    return vec, clf, metrics

def classify(vec, clf, texts):
    P = clf.predict_proba(vec.transform(texts))
    return clf.classes_[P.argmax(1)], P.max(1)

def apply_labels(df, vec, clf):
    df = df.copy(); df["category_source"] = np.where(df.category.notna(), "Recorded", "AI-classified")
    m = df.category.isna()
    if m.any():
        c, p = classify(vec, clf, df.loc[m, "complaint_text"]); df.loc[m, "category"] = c
    df["category"] = df["category"].astype(str)
    return df

# ---------- 2. Theme discovery (unsupervised) ----------
def themes(df, k=9):
    vec = TfidfVectorizer(stop_words="english", min_df=3, ngram_range=(1, 2), max_features=3000)
    X = vec.fit_transform(df.complaint_text.str.replace(r"Customer complaint:|Escalated by customer -|Retailer feedback:", "", regex=True))
    km = KMeans(n_clusters=k, n_init=5, random_state=3).fit(X)
    terms = np.array(vec.get_feature_names_out()); rows = []
    for c in range(k):
        idx = np.argsort(km.cluster_centers_[c])[::-1][:6]
        sub = df[km.labels_ == c]
        rows.append(dict(theme=c + 1, complaints=len(sub), top_terms=", ".join(terms[idx]),
                         dominant_category=sub.category.mode().iat[0], dominant_product=sub["product"].mode().iat[0],
                         last_12m=int((sub.date >= df.date.max() - pd.DateOffset(months=12)).sum())))
    return pd.DataFrame(rows).sort_values("complaints", ascending=False).reset_index(drop=True)

# ---------- 3. Pareto ----------
def pareto(df, col):
    t = df[col].dropna().value_counts().rename("complaints").to_frame()
    t["share_%"] = (t.complaints / t.complaints.sum() * 100).round(1); t["cumulative_%"] = t["share_%"].cumsum().round(1)
    return t.reset_index().rename(columns={"index": col})

# ---------- 4. Chronic / recurring ----------
def mark_repeats(df):
    d = df.sort_values("date").copy()
    key = ["product", "category", "root_cause"]; k = d.dropna(subset=["root_cause"])
    gap = k.groupby(key)["date"].diff().dt.days
    d["is_repeat"] = False; d.loc[k.index, "is_repeat"] = (gap <= 365).fillna(False)
    return d

def chronic_table(df, years=10):
    end = df.date.max(); recent = df[df.date > end - pd.DateOffset(years=years)]
    g = recent.groupby(["product", "category"])
    t = g.agg(complaints=("complaint_id", "size"),
              active_years=("date", lambda s: s.dt.year.nunique()),
              repeat_pct=("is_repeat", lambda s: round(s.mean() * 100, 1)),
              ineffective_pct=("action_effective", lambda s: round((s.dropna() == False).mean() * 100, 1) if s.notna().any() else np.nan)).reset_index()
    last12 = df[df.date > end - pd.DateOffset(months=12)].groupby(["product", "category"]).size().rename("last_12m")
    prev12 = df[(df.date > end - pd.DateOffset(months=24)) & (df.date <= end - pd.DateOffset(months=12))].groupby(["product", "category"]).size().rename("prev_12m")
    t = t.merge(last12, on=["product", "category"], how="left").merge(prev12, on=["product", "category"], how="left").fillna({"last_12m": 0, "prev_12m": 0})
    t["chronic_score"] = ((t.active_years / years) * 0.35 + (t.repeat_pct / 100) * 0.35 + (t.complaints / t.complaints.max()) * 0.3).round(3) * 100
    return t.sort_values("chronic_score", ascending=False).reset_index(drop=True)

def action_effectiveness(df, min_n=15):
    k = df.dropna(subset=["corrective_action", "action_effective"]).copy(); k["action_effective"] = k.action_effective.astype(bool)
    t = k.groupby("corrective_action").agg(times_used=("complaint_id", "size"), effective_pct=("action_effective", lambda s: round(s.mean() * 100, 1)),
                                           repeat_pct=("is_repeat", lambda s: round(s.mean() * 100, 1))).reset_index()
    t["verdict"] = np.where(t.effective_pct >= 75, "Proven", np.where(t.effective_pct >= 45, "Moderate", "Weak - avoid as sole action"))
    return t[t.times_used >= min_n].sort_values("effective_pct", ascending=False).reset_index(drop=True)

def action_by_root_cause(df, min_n=8):
    k = df.dropna(subset=["root_cause", "corrective_action", "action_effective"]).copy(); k["action_effective"] = k.action_effective.astype(bool)
    t = k.groupby(["root_cause", "corrective_action"]).agg(n=("complaint_id", "size"), effective_pct=("action_effective", lambda s: round(s.mean() * 100, 1))).reset_index()
    return t[t.n >= min_n]

# ---------- 5. Relationships ----------
def rate_matrix(df, mo, rows="product", cols="category", last_years=None):
    d = df if last_years is None else df[df.date > df.date.max() - pd.DateOffset(years=last_years)]
    return pd.crosstab(d[rows], d[cols])

# ---------- 6. Predictive risk model ----------
FEATS = ["lag1", "lag3", "lag12", "roll6", "moisture", "seal", "volume", "mon_sin", "mon_cos", "prod_code", "cat_code", "z_moist", "z_seal"]

def build_panel(df, mo):
    prods, cats = sorted(df["product"].unique()), sorted(df.category.unique())
    idx = pd.MultiIndex.from_product([prods, cats, pd.date_range(mo.month.min(), mo.month.max(), freq="MS")], names=["product", "category", "month"])
    cnt = df.groupby(["product", "category", "month"]).size().rename("n")
    p = cnt.reindex(idx, fill_value=0).reset_index()
    g = p.groupby(["product", "category"])["n"]
    p["lag1"] = g.shift(0); p["lag3"] = g.transform(lambda s: s.rolling(3, min_periods=1).sum())
    p["lag12"] = g.transform(lambda s: s.rolling(12, min_periods=1).sum()); p["roll6"] = g.transform(lambda s: s.rolling(6, min_periods=1).mean())
    p = p.merge(mo[["product", "month", "avg_moisture", "avg_seal_temp", "volume_tonnes"]], on=["product", "month"], how="left")
    p = p.rename(columns={"avg_moisture": "moisture", "avg_seal_temp": "seal", "volume_tonnes": "volume"})
    for c, z in (("moisture", "z_moist"), ("seal", "z_seal")):
        gg = p.groupby("product")[c]
        p[z] = (p[c] - gg.transform(lambda s: s.rolling(36, min_periods=12).mean())) / gg.transform(lambda s: s.rolling(36, min_periods=12).std())
    p["z_moist"] = p.z_moist.fillna(0); p["z_seal"] = p.z_seal.fillna(0)
    p["mon_sin"] = np.sin(2 * np.pi * (p.month.dt.month + 1) / 12); p["mon_cos"] = np.cos(2 * np.pi * (p.month.dt.month + 1) / 12)  # features describe NEXT month
    p["prod_code"] = p["product"].map({v: i for i, v in enumerate(prods)}); p["cat_code"] = p.category.map({v: i for i, v in enumerate(cats)})
    p["target"] = (p.groupby(["product", "category"])["n"].shift(-1) > 0).astype(float)
    p.loc[p.groupby(["product", "category"]).tail(1).index, "target"] = np.nan
    return p

def train_risk(df, mo, test_months=36):
    p = build_panel(df, mo); last = p.month.max()
    labelled = p.dropna(subset=["target"]); cut = last - pd.DateOffset(months=test_months)
    tr, te = labelled[labelled.month <= cut], labelled[labelled.month > cut]
    mk = lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=.06, max_iter=250, l2_regularization=1.0, random_state=0, categorical_features=[FEATS.index("prod_code"), FEATS.index("cat_code")])
    m = mk().fit(tr[FEATS], tr.target); pt = m.predict_proba(te[FEATS])[:, 1]
    base = tr.target.mean()
    metrics = dict(auc=roc_auc_score(te.target, pt), brier=brier_score_loss(te.target, pt), brier_naive=brier_score_loss(te.target, np.full(len(te), base)),
                   n_train=len(tr), n_test=len(te), test_from=str((cut + pd.DateOffset(months=1)).date()))
    m = mk().fit(labelled[FEATS], labelled.target)    # final model on all history
    cur = p[p.month == last].copy(); cur["risk_prob"] = m.predict_proba(cur[FEATS])[:, 1]
    # baseline = product-category average risk over previous 24 months for comparison
    hist = p[(p.month > last - pd.DateOffset(months=24)) & (p.month <= last - pd.DateOffset(months=1))]
    hist = hist.assign(prob=m.predict_proba(hist[FEATS])[:, 1]).groupby(["product", "category"]).prob.mean().rename("baseline_prob")
    cur = cur.merge(hist, on=["product", "category"]); cur["uplift"] = (cur.risk_prob / cur.baseline_prob).round(2)
    cur["forecast_month"] = (last + pd.DateOffset(months=1)).strftime("%b %Y")
    return m, metrics, cur[["product", "category", "forecast_month", "risk_prob", "baseline_prob", "uplift", "z_moist", "z_seal", "lag3", "lag12"]], p

def risk_band(prob, uplift):
    if prob >= .55 or (uplift >= 1.6 and prob >= .35): return "High"
    if prob >= .35 or uplift >= 1.3: return "Medium"
    return "Low"

def product_risk(cur):
    c = cur.copy(); c["band"] = [risk_band(a, b) for a, b in zip(c.risk_prob, c.uplift)]
    g = c.groupby("product").apply(lambda x: pd.Series(dict(
        any_complaint_prob=1 - np.prod(1 - x.risk_prob), top_category=x.sort_values("risk_prob").category.iat[-1],
        top_category_prob=x.risk_prob.max(), high_risk_categories=int((x.band == "High").sum()), max_uplift=x.uplift.max())), include_groups=False)
    return g.sort_values("any_complaint_prob", ascending=False).reset_index()

def early_warnings(cur):
    """Process-parameter drift linked to the complaint category it drives."""
    rows = []
    for _, r in cur.iterrows():
        if r.category == "Infestation" and r.z_moist > 1.5:
            rows.append(dict(product=r["product"], indicator="Grain moisture drift", z_score=round(r.z_moist, 1), linked_category=r.category, action="Check dryer + storage moisture; inspect stock before dispatch"))
        if r.category == "Packaging Issue" and r.z_seal < -1.5:
            rows.append(dict(product=r["product"], indicator="Sealing temperature low", z_score=round(r.z_seal, 1), linked_category=r.category, action="Verify sealing jaw temperature + dwell time"))
        if r.uplift >= 1.6 and r.lag3 >= 3:
            rows.append(dict(product=r["product"], indicator="Complaint cluster (3-month surge)", z_score=round(r.uplift, 1), linked_category=r.category, action="Open preventive CAPA before next dispatch cycle"))
    return pd.DataFrame(rows).drop_duplicates()

# ---------- 7. Recommendations ----------
GENERIC = {
    "Infestation": ["Tighten moisture limit at drying/storage release", "Pest-control audit of warehouses and transport", "Check seal integrity on retained samples"],
    "Contamination / Foreign Matter": ["Verify sieve mesh and magnet strength", "Run metal-detector challenge test every shift", "Review maintenance debris control"],
    "Packaging Issue": ["Verify sealing temperature and dwell time", "Re-qualify film lot and seal-strength tests", "Check coding/print verification"],
    "Product Quality Deviation": ["Tighten incoming raw-material specs", "Verify blend ratio records", "Add in-process quality checks"],
    "Physical Defect": ["Control moisture and cooling before pack", "Inspect milling rolls", "Review conveying breakage points"],
    "Sensory Characteristics": ["Review storage dwell time (FIFO)", "Check roasting/process parameters", "Verify packaging barrier"],
    "Weight / Quantity": ["Calibrate filler and check-weigher", "Enable auto-reject limits", "Audit set-up checks"],
    "Process-Related Defect": ["Verify change-over cleaning", "Revalidate mixing time", "Dispatch label double-check"],
    "Other": ["Review documentation and delivery planning"]}

def likely_root_causes(df, product, category, top=4):
    s = df[(df["product"] == product) & (df.category == category)].dropna(subset=["root_cause"])
    recent = s[s.date > df.date.max() - pd.DateOffset(years=5)]
    s = recent if len(recent) >= 15 else s
    if len(s) < 5: s = df[df.category == category].dropna(subset=["root_cause"])
    t = s.root_cause.value_counts(normalize=True).head(top).mul(100).round(1)
    return t.rename("likelihood_%").reset_index()

def recommend(df, abr, product, category, top_rc=3):
    rc = likely_root_causes(df, product, category, top_rc); out = []
    for _, r in rc.iterrows():
        a = abr[abr.root_cause == r.root_cause].sort_values("effective_pct", ascending=False)
        for _, x in a.iterrows():
            out.append(dict(likely_root_cause=r.root_cause, likelihood_pct=r["likelihood_%"], action=x.corrective_action, historical_effectiveness_pct=x.effective_pct,
                            recommendation="Prioritise" if x.effective_pct >= 75 else ("Use with support" if x.effective_pct >= 45 else "Avoid as sole action")))
    return pd.DataFrame(out), GENERIC.get(category, [])

# ---------- 8. KPIs ----------
def kpis(df, mo):
    end = df.date.max(); y = lambda a, b: df[(df.date > end - pd.DateOffset(months=a)) & (df.date <= end - pd.DateOffset(months=b))]
    vol = lambda a, b: mo[(mo.month > end - pd.DateOffset(months=a)) & (mo.month <= end - pd.DateOffset(months=b))].volume_tonnes.sum()
    cur, prev = y(12, 0), y(24, 12); base = df[df.date < df.date.min() + pd.DateOffset(years=3)]
    rate = lambda d, v: len(d) / v * 100
    return dict(
        complaints_12m=len(cur), complaints_prev=len(prev), rate_12m=rate(cur, vol(12, 0)), rate_prev=rate(prev, vol(24, 12)),
        rate_baseline=rate(base, mo[mo.month < mo.month.min() + pd.DateOffset(years=3)].volume_tonnes.sum()),
        critical_12m=int((cur.severity == "Critical").sum()), critical_prev=int((prev.severity == "Critical").sum()),
        repeat_pct=cur.is_repeat.mean() * 100, repeat_prev=prev.is_repeat.mean() * 100,
        effective_pct=cur.action_effective.dropna().astype(bool).mean() * 100, closure_days=cur.closure_days.mean(), closure_prev=prev.closure_days.mean())

def annual_rate(df, mo):
    a = df.groupby(df.month.dt.year).size().rename("complaints"); v = mo.groupby(mo.month.dt.year).volume_tonnes.sum().rename("tonnes")
    nm = df.groupby(df.month.dt.year).month.nunique().rename("nm"); t = pd.concat([a, v, nm], axis=1).dropna(); t = t[t.nm == 12].drop(columns="nm")   # full calendar years only
    t["per_100t"] = t.complaints / t.tonnes * 100; return t.reset_index().rename(columns={"month": "year"})


# ================= Item 4 upgrade: relationships =================
def lift_table(df, row, col, min_n=20, top=12):
    """lift = P(col | row) / P(col): how much more often this row value sees this col value than average."""
    d = df.dropna(subset=[row, col]); pc = d[col].value_counts(normalize=True)
    g = d.groupby([row, col]).size().rename("n").reset_index(); tot = d.groupby(row).size().rename("row_total")
    g = g.merge(tot, on=row); g["share_%"] = (g.n / g.row_total * 100).round(1); g["lift"] = (g.n / g.row_total / g[col].map(pc)).round(2)
    return g[g.n >= min_n].sort_values("lift", ascending=False).head(top).reset_index(drop=True)

def parameter_bands(df, mo, category, param, bins=5):
    """Complaint rate (per 100 t) of one category across bands of a process parameter, product-month level."""
    n = df[df.category == category].groupby(["product", "month"]).size().rename("n")
    m = mo.merge(n.reset_index(), on=["product", "month"], how="left").fillna({"n": 0})
    m["band"] = pd.qcut(m[param], bins, duplicates="drop")
    t = m.groupby("band", observed=True).agg(complaints=("n", "sum"), tonnes=("volume_tonnes", "sum"), months=("n", "size")).reset_index()
    t["per_100t"] = (t.complaints / t.tonnes * 100).round(3); t["band"] = t.band.map(lambda i: f"{i.left:.1f}-{i.right:.1f}")
    return t

# ================= Item 5 upgrade: process-level risk =================
ROOT_AREA = {
 "High grain moisture at storage": "Drying & storage", "Warehouse pest-control lapse": "Drying & storage", "Long dwell time in distribution": "Warehouse & logistics",
 "Infested raw material received": "Raw material", "Raw-material cleaning inadequate": "Raw material", "Raw-material variability": "Raw material", "Raw-material odour carry-over": "Raw material",
 "Sieve / magnet failure": "Cleaning & foreign-matter control", "Maintenance debris (metal)": "Maintenance", "Machine jaw wear": "Maintenance", "Milling roll wear": "Maintenance",
 "Personnel hygiene lapse": "People & hygiene", "Operator set-up error": "People & hygiene", "Sieving lapse": "Cleaning & foreign-matter control",
 "Pack seal breach allowing entry": "Packaging & sealing", "Low sealing temperature": "Packaging & sealing", "Packaging film supplier variation": "Packaging & sealing",
 "Packing-material contamination": "Packaging & sealing", "Packaging barrier failure": "Packaging & sealing", "Coding / printing error": "Packaging & sealing",
 "Carton handling damage in transit": "Warehouse & logistics", "Incorrect blend ratio": "Blending & process control", "Process parameter drift": "Blending & process control",
 "Mixing non-uniformity": "Blending & process control", "Over / under roasting": "Blending & process control", "Improper cooling before pack": "Blending & process control",
 "Lab testing gap": "Quality control", "Moisture caking": "Drying & storage", "Conveying breakage": "Maintenance", "Rancidity due to storage": "Warehouse & logistics",
 "Filler calibration drift": "Filling & weighing", "Check-weigher bypass": "Filling & weighing", "Moisture loss in storage": "Drying & storage",
 "Wrong grade / label dispatched": "Dispatch & change-over", "Changeover cleaning lapse": "Dispatch & change-over",
 "Documentation / invoicing error": "Dispatch & change-over", "Delivery delay": "Warehouse & logistics", "Customer service gap": "Dispatch & change-over"}

def _root_lik(df, years=5, recent_months=12, min_recent=8):
    """P(root cause | product, category). Uses the last 12 months where there is enough evidence (captures new shifts), else the last 5 years."""
    end = df.date.max(); d = df.dropna(subset=["root_cause"])
    def share(x): return x.groupby(["product", "category"]).root_cause.value_counts(normalize=True).rename("lk").reset_index()
    long_ = share(d[d.date > end - pd.DateOffset(years=years)])
    rec = d[d.date > end - pd.DateOffset(months=recent_months)]; cnt = rec.groupby(["product", "category"]).size().rename("n").reset_index()
    rec_s = share(rec).merge(cnt, on=["product", "category"]); rec_s = rec_s[rec_s.n >= min_recent].drop(columns="n")
    return long_, rec_s

def process_risk(df, cur):
    """Which PROCESS failure modes will drive next month's complaints?
    risk_index = sum over product x category of P(complaint next month) x P(root cause | product, category)."""
    long_, rec = _root_lik(df); keys = ["product", "category"]
    now_lk = long_.merge(rec, on=keys + ["root_cause"], how="outer", suffixes=("_l", "_r"))
    has_rec = now_lk.groupby(keys).lk_r.transform(lambda x: x.notna().any())
    now_lk["lk_now"] = np.where(has_rec, now_lk.lk_r.fillna(0), now_lk.lk_l.fillna(0)); now_lk["lk_usual"] = now_lk.lk_l.fillna(0)
    m = cur.merge(now_lk, on=keys)
    m["now"] = m.risk_prob * m.lk_now; m["usual"] = m.baseline_prob * m.lk_usual
    t = m.groupby("root_cause").agg(risk_index=("now", "sum"), usual_index=("usual", "sum")).reset_index()
    t["vs_usual_x"] = (t.risk_index / t.usual_index).round(2); t["risk_index"] = t.risk_index.round(2)
    t["process_area"] = t.root_cause.map(ROOT_AREA).fillna("Other")
    t["band"] = np.where(t.vs_usual_x >= 1.4, "High", np.where(t.vs_usual_x >= 1.15, "Medium", "Low"))
    return t.drop(columns="usual_index").sort_values("risk_index", ascending=False).reset_index(drop=True)

def area_risk(pr):
    a = pr.groupby("process_area").agg(risk_index=("risk_index", "sum"), top_failure_mode=("root_cause", "first"), max_vs_usual=("vs_usual_x", "max")).reset_index()
    return a.sort_values("risk_index", ascending=False).reset_index(drop=True)

def line_risk(df, cur):
    """Production-line risk: expected number of complaint events next month (sum of category probabilities) vs the line's usual level."""
    line = df.drop_duplicates("product").set_index("product").line
    c = cur.assign(line=cur["product"].map(line)); rows = []
    for ln, g in c.groupby("line"):
        top = g.sort_values("risk_prob").iloc[-1]; hot = g.sort_values("uplift").iloc[-1]
        rows.append(dict(line=ln, products=", ".join(sorted(g["product"].unique())), expected_events=g.risk_prob.sum(), usual_events=g.baseline_prob.sum(),
                         top_risk=f"{top['product']} · {top.category}", top_risk_prob=top.risk_prob,
                         hottest_signal=f"{hot['product']} · {hot.category}", hottest_uplift=hot.uplift, high_cells=int((g.band == "High").sum())))
    t = pd.DataFrame(rows); t["vs_usual_x"] = (t.expected_events / t.usual_events).round(2)
    t["expected_events"] = t.expected_events.round(2); t["usual_events"] = t.usual_events.round(2)
    return t.sort_values("vs_usual_x", ascending=False).reset_index(drop=True)
