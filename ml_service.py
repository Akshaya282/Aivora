"""Live ML service for the Zero Customer Complaint system.

Trains every model once (and again after each new complaint), and answers:
  - dashboard data (everything the HTML renders, from the real models)
  - classify(text)            -> category + confidence for a free-text complaint
  - predict(product, cat)     -> next-month complaint probability with a bootstrap confidence interval (+ what-if)
  - metrics                   -> accuracy, sensitivity, SPECIFICITY, confusion matrices, AUC with CI, calibration
"""
import threading, time, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
import engine as E
import build_html as B

N_BOOT = 15          # bootstrap models for the confidence interval
SEED = 7


def _rates(cm_):
    """Per-class sensitivity (recall), specificity, precision from a one-vs-rest view of a multiclass confusion matrix."""
    n = cm_.sum(); out = []
    for k in range(cm_.shape[0]):
        tp = cm_[k, k]; fn = cm_[k].sum() - tp; fp = cm_[:, k].sum() - tp; tn = n - tp - fn - fp
        out.append(dict(sensitivity=tp / (tp + fn) if tp + fn else np.nan, specificity=tn / (tn + fp) if tn + fp else np.nan, precision=tp / (tp + fp) if tp + fp else np.nan, support=int(tp + fn)))
    return out


def _alert_threshold(y, p, min_spec=0.70):
    """Smallest alert threshold whose false-alarm rate is acceptable (specificity >= min_spec) on VALIDATION data, which maximises sensitivity under that limit."""
    for t in np.linspace(.1, .95, 86):
        pred = p >= t; tn = ((~pred) & (y == 0)).sum(); fp = ((pred) & (y == 0)).sum()
        if tn / max(tn + fp, 1) >= min_spec: return float(t)
    return .95


def _op_point(y, p, t):
    pred = p >= t; tp = int(((pred) & (y == 1)).sum()); fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum()); fp = int(((pred) & (y == 0)).sum())
    return dict(threshold=round(float(t), 2), sensitivity=round(tp / max(tp + fn, 1), 3), specificity=round(tn / max(tn + fp, 1), 3), precision=round(tp / max(tp + fp, 1), 3), alerts_share=round(float(pred.mean()), 3))


def _conf_label(width):
    return "High" if width <= .12 else ("Medium" if width <= .25 else "Low")


class Service:
    def __init__(self):
        self.lock = threading.Lock(); self.ready = False; self.train_seconds = 0.0; self.train()

    # ------------------------------------------------------------------ training
    def train(self):
        t0 = time.time(); df, mo = E.load()
        # ---- text classifier (with an honest held-out evaluation)
        lab = df[df.category.notna()]
        Xtr, Xte, ytr, yte = train_test_split(lab.complaint_text, lab.category, test_size=.2, random_state=1, stratify=lab.category)
        ev = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=2000, C=8)).fit(Xtr, ytr)
        P = ev.predict_proba(Xte); classes = ev.classes_; pred = classes[P.argmax(1)]; conf = P.max(1)
        cmx = confusion_matrix(yte, pred, labels=classes); rates = _rates(cmx)
        cv = cross_val_score(make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=2000, C=8)),
                             lab.complaint_text, lab.category, cv=StratifiedKFold(5, shuffle=True, random_state=1), n_jobs=1)
        hi, lo = conf >= .8, conf < .55
        clf_perf = dict(accuracy=accuracy_score(yte, pred), macro_f1=f1_score(yte, pred, average="macro"), n_test=int(len(yte)), cv_mean=float(cv.mean()), cv_std=float(cv.std()),
                        macro_sensitivity=float(np.nanmean([r["sensitivity"] for r in rates])), macro_specificity=float(np.nanmean([r["specificity"] for r in rates])),
                        classes=list(classes), confusion=cmx.tolist(), per_class=[dict(cls=c, **{k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 4) if k != "support" else v) for k, v in r.items()}) for c, r in zip(classes, rates)],
                        acc_high_conf=float((pred[hi] == yte.values[hi]).mean()) if hi.any() else None, share_high_conf=float(hi.mean()),
                        acc_low_conf=float((pred[lo] == yte.values[lo]).mean()) if lo.any() else None, share_low_conf=float(lo.mean()))
        # ---- production text model on all labelled data
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(lab.complaint_text)
        clf = LogisticRegression(max_iter=2000, C=8).fit(vec.transform(lab.complaint_text), lab.category)
        df = E.mark_repeats(E.apply_labels(df, vec, clf))
        cm_old = dict(accuracy=clf_perf["accuracy"], macro_f1=clf_perf["macro_f1"], n_test=clf_perf["n_test"])
        # ---- risk model: time-based test, threshold picked on a validation year (never on the test set)
        p = E.build_panel(df, mo); labelled = p.dropna(subset=["target"]); last = p.month.max(); cut = last - pd.DateOffset(months=36); vcut = cut - pd.DateOffset(months=12)
        cat_idx = [E.FEATS.index("prod_code"), E.FEATS.index("cat_code")]
        mk = lambda seed=0: HistGradientBoostingClassifier(max_depth=4, learning_rate=.06, max_iter=250, l2_regularization=1.0, random_state=seed, categorical_features=cat_idx)
        tr, te = labelled[labelled.month <= cut], labelled[labelled.month > cut]
        sub, val = labelled[labelled.month <= vcut], labelled[(labelled.month > vcut) & (labelled.month <= cut)]
        thr = _alert_threshold(val.target.values, mk().fit(sub[E.FEATS], sub.target).predict_proba(val[E.FEATS])[:, 1])
        m_te = mk().fit(tr[E.FEATS], tr.target); pt = m_te.predict_proba(te[E.FEATS])[:, 1]; y = te.target.values.astype(int); yp = (pt >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        rng = np.random.default_rng(SEED); aucs = []
        for _ in range(300):
            ix = rng.integers(0, len(y), len(y))
            if y[ix].min() != y[ix].max(): aucs.append(roc_auc_score(y[ix], pt[ix]))
        bins = pd.cut(pt, [0, .2, .35, .5, .65, .8, 1.0], include_lowest=True); cal = pd.DataFrame(dict(b=bins, p=pt, y=y)).groupby("b", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
        percat = []
        for c, g in te.assign(pt=pt).groupby("category"):
            if g.target.nunique() == 2: percat.append(dict(category=c, auc=round(float(roc_auc_score(g.target, g.pt)), 3), base_rate=round(float(g.target.mean()), 3), n=int(len(g))))
        risk_perf = dict(auc=float(roc_auc_score(y, pt)), auc_lo=float(np.percentile(aucs, 5)), auc_hi=float(np.percentile(aucs, 95)), threshold=thr, sensitivity=float(tp / max(tp + fn, 1)), specificity=float(tn / max(tn + fp, 1)),
                         precision=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)), accuracy=float((tp + tn) / len(y)), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), n_test=int(len(y)), base_rate=float(y.mean()),
                         operating_points=[_op_point(y, pt, t) for t in sorted({round(thr, 2), .3, .4, .5, .6, .7})], min_spec_target=0.70,
                         brier=float(np.mean((pt - y) ** 2)), brier_naive=float(np.mean((tr.target.mean() - y) ** 2)), test_from=str((cut + pd.DateOffset(months=1)).date()),
                         calibration=[dict(pred=round(float(r.pred), 3), obs=round(float(r.obs), 3), n=int(r.n), label=str(r.b)) for r in cal.itertuples()], per_category=percat)
        # ---- production risk model + bootstrap ensemble for confidence intervals
        model, rm, cur, panel = E.train_risk(df, mo); cur["band"] = [E.risk_band(a, b) for a, b in zip(cur.risk_prob, cur.uplift)]
        boots = []
        for b in range(N_BOOT):
            ix = np.random.default_rng(SEED + b).integers(0, len(labelled), len(labelled)); s = labelled.iloc[ix]; boots.append(mk(b).fit(s[E.FEATS], s.target))
        feat_now = panel[panel.month == panel.month.max()].reset_index(drop=True)
        self.__dict__.update(df=df, mo=mo, vec=vec, clf=clf, cm_old=cm_old, model=model, cur=cur, panel=panel, feat_now=feat_now, boots=boots, thr=thr, calib=cal,
                             clf_perf=clf_perf, risk_perf=risk_perf, classes=list(classes))
        self.pred_cache = {}
        self._ens = self._ensemble(feat_now)
        self.ready = True; self.train_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ prediction
    def _ensemble(self, X):
        M = np.column_stack([m.predict_proba(X[E.FEATS])[:, 1] for m in self.boots])
        return dict(mean=M.mean(1), lo=np.percentile(M, 5, axis=1), hi=np.percentile(M, 95, axis=1), agree=(M >= self.thr).mean(1))

    def _hit_rate(self, prob):
        c = self.calib; i = int((c.pred - prob).abs().idxmin()); r = c.loc[i]
        return dict(said=round(float(r.pred), 3), happened=round(float(r.obs), 3), n=int(r.n))

    def predict(self, product, category, dz_moist=0.0, dz_seal=0.0):
        with self.lock:
            row = self.feat_now[(self.feat_now["product"] == product) & (self.feat_now.category == category)]
            if row.empty: raise KeyError("unknown product/category")
            r = row.copy(); sd_m = self.mo[self.mo["product"] == product].avg_moisture.std(); sd_s = self.mo[self.mo["product"] == product].avg_seal_temp.std()
            r["z_moist"] += dz_moist; r["moisture"] += dz_moist * sd_m; r["z_seal"] += dz_seal; r["seal"] += dz_seal * sd_s
            prob = float(self.model.predict_proba(r[E.FEATS])[:, 1][0]); e = self._ensemble(r)
            base = float(self.cur[(self.cur["product"] == product) & (self.cur.category == category)].baseline_prob.iat[0])
            lo, hi = float(min(e["lo"][0], prob)), float(max(e["hi"][0], prob)); w = hi - lo
            return dict(product=product, category=category, probability=round(prob, 4), ci_low=round(lo, 4), ci_high=round(hi, 4), ci_level="90% bootstrap interval", confidence=_conf_label(w),
                        model_agreement=round(float(e["agree"][0]), 3), decision="Alert" if prob >= self.thr else "No alert", threshold=round(self.thr, 3), usual=round(base, 3), vs_usual=round(prob / base, 2),
                        band=E.risk_band(prob, prob / base), track_record=self._hit_rate(prob), what_if=dict(moisture_sd=dz_moist, seal_sd=dz_seal), forecast_month=str(self.cur.forecast_month.iat[0]))

    def classify(self, text):
        with self.lock:
            P = self.clf.predict_proba(self.vec.transform([text]))[0]; order = np.argsort(P)[::-1]; top = float(P[order[0]]); cp = self.clf_perf
            lvl = "High" if top >= .8 else ("Medium" if top >= .55 else "Low")
            note = {"High": f"On held-out complaints, predictions at this confidence were right {cp['acc_high_conf']:.0%} of the time.", "Medium": "Plausible, but please confirm the category.",
                    "Low": f"Low confidence: send to a person for review (accuracy below 55% confidence was only {cp['acc_low_conf']:.0%} on held-out complaints)." if cp["acc_low_conf"] is not None else "Low confidence: send to a person for review."}[lvl]
            return dict(category=str(self.classes[order[0]]), confidence=round(top, 4), confidence_level=lvl, note=note,
                        top3=[dict(category=str(self.classes[i]), probability=round(float(P[i]), 4)) for i in order[:3]], needs_review=lvl == "Low")

    # ------------------------------------------------------------------ dashboard payload
    def data(self):
        with self.lock:
            e = self._ens; extra = dict(perf=dict(classifier=self.clf_perf, risk=self.risk_perf, boot_models=N_BOOT, trained_seconds=self.train_seconds), live=True)
            d = B.make_data(self.df, self.mo, self.vec, self.clf, self.cm_old, self.model, dict(auc=self.risk_perf["auc"], brier=self.risk_perf["brier"], brier_naive=self.risk_perf["brier_naive"], n_test=self.risk_perf["n_test"], test_from=self.risk_perf["test_from"]), self.cur, extra)
            ip = {p: i for i, p in enumerate(d["prods"])}; ic = {c: i for i, c in enumerate(d["cats"])}
            by = {(ip[r["product"]], ic[r.category]): i for i, r in self.feat_now.iterrows()}
            for r in d["risk"]:
                i = by[(r["p"], r["c"])]; lo, hi = float(min(e["lo"][i], r["prob"])), float(max(e["hi"][i], r["prob"]))
                r.update(lo=round(lo, 3), hi=round(hi, 3), conf=_conf_label(hi - lo), agree=round(float(e["agree"][i]), 3), alert=bool(r["prob"] >= self.thr))
            d["thr"] = round(self.thr, 3)
            return d

    # ------------------------------------------------------------------ learning from new complaints
    def add_complaint(self, rec):
        raw = pd.read_csv(E.DATA); line = self.df[self.df["product"] == rec["product"]].line.iat[0]
        row = dict(complaint_id=f"CMP-{len(raw)+1:05d}", date=rec["date"], customer=rec["customer"], product=rec["product"], line=line, batch_no=rec.get("batch_no") or np.nan,
                   complaint_text=rec["text"], category=rec["category"], severity=rec.get("severity", "Major"))
        pd.concat([raw, pd.DataFrame([row])], ignore_index=True).to_csv(E.DATA, index=False)
        self.train(); return row["complaint_id"]
