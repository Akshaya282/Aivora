"""Zero Customer Complaint Analytics & Prediction System - ABT-CPD prototype (Team Aivora)."""
import os, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import engine as E

st.set_page_config(page_title="Zero Complaint System", page_icon="🎯", layout="wide")
BLUE, GREY, RED = "#2a78d6", "#9aa0a6", "#e34948"
CAT_COL = {"Infestation": "#2a78d6", "Packaging Issue": "#eb6834", "Contamination / Foreign Matter": "#1baf7a", "Product Quality Deviation": "#eda100",
           "Physical Defect": "#e87ba4", "Sensory Characteristics": "#008300", "Weight / Quantity": "#6250d6", "Process-Related Defect": "#e34948", "Other": "#9aa0a6"}
SEQ = [[0, "#eef4fc"], [1, "#12417f"]]
LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))

@st.cache_resource(show_spinner="Training models on complaint history...")
def pipeline(mtime):
    df, mo = E.load(); vec, clf, cm = E.train_classifier(df)
    df = E.mark_repeats(E.apply_labels(df, vec, clf))
    model, rm, cur, panel = E.train_risk(df, mo)
    cur["band"] = [E.risk_band(a, b) for a, b in zip(cur.risk_prob, cur.uplift)]
    return dict(df=df, mo=mo, vec=vec, clf=clf, cm=cm, rm=rm, cur=cur, themes=E.themes(df), chronic=E.chronic_table(df), actions=E.action_effectiveness(df),
                abr=E.action_by_root_cause(df), kpi=E.kpis(df, mo), annual=E.annual_rate(df, mo), prisk=E.product_risk(cur), ew=E.early_warnings(cur),
                proc=(pr := E.process_risk(df, cur)), area=E.area_risk(pr), lines=E.line_risk(df, cur),
                b_inf=E.parameter_bands(df, mo, 'Infestation', 'avg_moisture'), b_pack=E.parameter_bands(df, mo, 'Packaging Issue', 'avg_seal_temp'))

P = pipeline(os.path.getmtime(E.DATA)); df_all, mo = P["df"], P["mo"]

# ---------------- sidebar filters ----------------
st.sidebar.title("🎯 Zero Complaint System")
st.sidebar.caption("ABT-CPD · Team Aivora · prototype on synthetic data")
y0, y1 = int(df_all.date.dt.year.min()), int(df_all.date.dt.year.max())
yr = st.sidebar.slider("Years", y0, y1, (y0, y1))
prods = st.sidebar.multiselect("Product", sorted(df_all["product"].unique()), default=[])
custs = st.sidebar.multiselect("Customer", sorted(df_all.customer.unique()), default=[])
f = df_all[(df_all.date.dt.year >= yr[0]) & (df_all.date.dt.year <= yr[1])]
if prods: f = f[f["product"].isin(prods)]
if custs: f = f[f.customer.isin(custs)]
st.sidebar.metric("Complaints in view", f"{len(f):,}"); st.sidebar.caption(f"{(df_all.category_source=='AI-classified').mean():.0%} of records had no category - classified by the NLP model "
                   f"(holdout accuracy {P['cm']['accuracy']:.1%}).")

st.title("Zero Customer Complaint · Analytics, Root Cause & Prediction")
tabs = st.tabs(["Overview & KPIs", "Pareto & Themes", "Trends", "Chronic & Actions", "Relationships", "Risk & Early Warning", "Recommendations", "Log new complaint"])

# ---------------- 1. overview ----------------
with tabs[0]:
    k = P["kpi"]; cur = P["cur"]
    hi = cur[cur.band == "High"].sort_values("risk_prob", ascending=False)
    if len(hi): st.error(f"**{len(hi)} high-risk product/complaint combinations for {cur.forecast_month.iat[0]}** - e.g. " + "; ".join(f"{r['product']} → {r.category} ({r.risk_prob:.0%})" for _, r in hi.head(3).iterrows()) + ". See *Risk & Early Warning*.")
    c = st.columns(5)
    c[0].metric("Complaints (last 12 m)", k["complaints_12m"], f"{k['complaints_12m']-k['complaints_prev']:+d} vs prior 12 m", delta_color="inverse")
    c[1].metric("Complaints per 100 t", f"{k['rate_12m']:.2f}", f"{(k['rate_12m']/k['rate_prev']-1)*100:+.1f}%", delta_color="inverse")
    c[2].metric("Critical complaints", k["critical_12m"], f"{k['critical_12m']-k['critical_prev']:+d}", delta_color="inverse")
    c[3].metric("Repeat complaints", f"{k['repeat_pct']:.0f}%", f"{k['repeat_pct']-k['repeat_prev']:+.1f} pts", delta_color="inverse")
    c[4].metric("Corrective actions effective", f"{k['effective_pct']:.0f}%")
    a = P["annual"]; last = a.iloc[-1]
    fut = pd.DataFrame({"year": [int(last.year) + i for i in range(0, 5)], "per_100t": [last.per_100t * (0.7 ** i) for i in range(0, 5)]})
    fig = go.Figure()
    fig.add_scatter(x=a.year, y=a.per_100t, mode="lines+markers", name="Actual", line=dict(color=BLUE, width=2), marker=dict(size=8), hovertemplate="%{x}: %{y:.2f} per 100 t<extra></extra>")
    fig.add_scatter(x=fut.year, y=fut.per_100t, mode="lines+markers", name="Glide-path: -30% per year toward zero", line=dict(color=RED, width=2, dash="dash"), marker=dict(size=8), hovertemplate="%{x}: target %{y:.2f}<extra></extra>")
    fig.update_layout(title="Complaints per 100 tonnes shipped - actual vs. glide-path to zero", yaxis_title="complaints / 100 t", **LAYOUT); st.plotly_chart(fig, width="stretch")
    st.subheader("KPI scorecard for moving towards zero")
    sc = pd.DataFrame([
        ["Complaints per 100 t shipped", f"{k['rate_12m']:.2f}", f"{k['rate_12m']*.7:.2f} (-30%/yr)", "Monthly"],
        ["Critical complaints (contamination / infestation)", k["critical_12m"], "0", "Monthly"],
        ["Repeat complaint share", f"{k['repeat_pct']:.0f}%", "< 20%", "Monthly"],
        ["Corrective-action effectiveness", f"{k['effective_pct']:.0f}%", "> 90%", "Quarterly"],
        ["Avg. complaint closure time (days)", f"{k['closure_days']:.1f}", "< 7", "Monthly"],
        ["Chronic product/complaint pairs flagged", int((P['chronic'].chronic_score > 80).sum()), "0", "Quarterly"],
        ["Early-warning lead time before complaint", "next month forecast", "≥ 30 days", "Monthly"]], columns=["KPI", "Current", "Target", "Review"]).astype(str)
    st.dataframe(sc, hide_index=True, width="stretch")

# ---------------- 2. pareto & themes ----------------
with tabs[1]:
    dim = st.selectbox("Pareto by", ["category", "product", "customer", "root_cause", "line", "corrective_action"], format_func=lambda s: s.replace("_", " ").title())
    t = E.pareto(f, dim).head(12)
    fig = go.Figure()
    fig.add_bar(x=t[dim], y=t["share_%"], name="Share of complaints (%)", marker_color=BLUE, hovertemplate="%{x}<br>%{y:.1f}% of complaints<extra></extra>")
    fig.add_scatter(x=t[dim], y=t["cumulative_%"], name="Cumulative share (%)", mode="lines+markers", line=dict(color="#eb6834", width=2), hovertemplate="%{x}<br>cumulative %{y:.1f}%<extra></extra>")
    fig.add_hline(y=80, line_dash="dot", line_color=GREY, annotation_text="80%")
    fig.update_layout(title=f"Pareto of complaints by {dim.replace('_',' ')}", yaxis=dict(title="% of complaints", range=[0, 102]), **LAYOUT); st.plotly_chart(fig, width="stretch")
    n80 = int((t["cumulative_%"] < 80).sum()) + 1
    st.caption(f"The top **{n80}** {dim.replace('_',' ')} values account for ~80% of the complaints in view.")
    st.subheader("Complaint themes discovered by unsupervised clustering (TF-IDF + K-Means)")
    st.dataframe(P["themes"], hide_index=True, width="stretch")

# ---------------- 3. trends ----------------
with tabs[2]:
    m = f.groupby(["month", "category"]).size().unstack(fill_value=0).asfreq("MS", fill_value=0)
    roll = m.rolling(12).sum().dropna(how="all"); roll = roll.reset_index().melt("month", var_name="category", value_name="rolling_12m")
    fig = px.line(roll, x="month", y="rolling_12m", color="category", color_discrete_map=CAT_COL, title="Rolling 12-month complaints by category")
    fig.update_traces(line_width=2); fig.update_layout(yaxis_title="complaints (12 m)", **LAYOUT); st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    s = f.groupby([f.date.dt.month, "category"]).size().unstack(fill_value=0).reindex(range(1, 13), fill_value=0)
    fig = px.imshow(s.T, color_continuous_scale=SEQ, aspect="auto", labels=dict(x="Month", y="", color="Complaints"), title="Seasonality: complaints by calendar month")
    fig.update_xaxes(tickvals=list(range(12)), ticktext=list("JFMAMJJASOND")); fig.update_layout(**LAYOUT); a.plotly_chart(fig, width="stretch")
    sv = f.groupby([f.date.dt.year, "severity"]).size().rename("n").reset_index()
    fig = px.bar(sv, x="date", y="n", color="severity", title="Severity mix by year", color_discrete_map={"Critical": RED, "Major": "#eda100", "Minor": GREY})
    fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=2); fig.update_layout(xaxis_title="", yaxis_title="complaints", **LAYOUT); b.plotly_chart(fig, width="stretch")

# ---------------- 4. chronic & actions ----------------
with tabs[3]:
    ch = P["chronic"].copy(); ch["flag"] = np.where(ch.chronic_score >= 80, "CHRONIC", "")
    st.subheader("Chronic / recurring problem areas (last 10 years)")
    st.caption("Chronic score = persistence across years (35%) + repeat-complaint rate (35%) + volume (30%). Repeat = same product, category and root cause again within 12 months.")
    st.dataframe(ch.head(15), hide_index=True, width="stretch", column_config={"chronic_score": st.column_config.ProgressColumn("chronic score", min_value=0, max_value=100, format="%.0f")})
    st.subheader("Which corrective actions actually work?")
    ac = P["actions"]
    fig = px.bar(ac.sort_values("effective_pct"), x="effective_pct", y="corrective_action", color="verdict", orientation="h", title="Historical effectiveness of corrective actions (%)",
                 color_discrete_map={"Proven": BLUE, "Moderate": GREY, "Weak - avoid as sole action": RED}, hover_data=["times_used", "repeat_pct"])
    fig.update_layout(yaxis_title="", xaxis_title="% of cases where action was effective", height=520, **LAYOUT); st.plotly_chart(fig, width="stretch")
    weak = ac[ac.verdict.str.startswith("Weak")]
    st.warning("**Ineffective actions still in routine use:** " + ", ".join(f"{r.corrective_action} ({r.times_used}x, {r.effective_pct:.0f}% effective)" for _, r in weak.iterrows()))

# ---------------- 5. relationships ----------------
with tabs[4]:
    a, b = st.columns(2)
    pc = pd.crosstab(f["product"], f.category)
    a.plotly_chart(px.imshow(pc, color_continuous_scale=SEQ, aspect="auto", text_auto=True, title="Product × complaint type", labels=dict(color="Complaints")).update_layout(**LAYOUT), width="stretch")
    cc = pd.crosstab(f.customer, f.category)
    b.plotly_chart(px.imshow(cc, color_continuous_scale=SEQ, aspect="auto", text_auto=True, title="Customer × complaint type", labels=dict(color="Complaints")).update_layout(**LAYOUT), width="stretch")
    st.subheader("Product → complaint type → root cause")
    top_n = st.slider("Show top flows", 10, 40, 22)
    s = f.dropna(subset=["root_cause"]).groupby(["product", "category", "root_cause"]).size().rename("n").reset_index().nlargest(top_n, "n")
    nodes = list(pd.unique(pd.concat([s["product"], s.category, s.root_cause]))); ix = {n: i for i, n in enumerate(nodes)}
    l1 = s.groupby(["product", "category"]).n.sum().reset_index(); l2 = s.groupby(["category", "root_cause"]).n.sum().reset_index()
    src = [ix[x] for x in l1["product"]] + [ix[x] for x in l2.category]; tgt = [ix[x] for x in l1.category] + [ix[x] for x in l2.root_cause]; val = list(l1.n) + list(l2.n)
    fig = go.Figure(go.Sankey(node=dict(label=nodes, pad=14, thickness=14, color=[CAT_COL.get(n, "#52514e") if n in CAT_COL else ("#c9d6e8" if n in set(s["product"]) else "#e6e3dc") for n in nodes]),
                              link=dict(source=src, target=tgt, value=val, color="rgba(120,130,150,.25)")))
    fig.update_layout(height=520, **LAYOUT); st.plotly_chart(fig, width="stretch")
    st.subheader("Process parameters vs complaints (does the process explain the complaints?)")
    st.caption("Complaint rate per 100 t shipped, split into five equal-size bands of the process parameter (product-month level, full history).")
    c1, c2 = st.columns(2)
    bi, bp = P["b_inf"], P["b_pack"]
    c1.plotly_chart(px.bar(bi, x="band", y="per_100t", title="Infestation complaints per 100 t by grain moisture band (%)", color_discrete_sequence=[BLUE], hover_data=["complaints", "months"]).update_layout(xaxis_title="average moisture band (%)", yaxis_title="complaints / 100 t", **LAYOUT), width="stretch")
    c2.plotly_chart(px.bar(bp, x="band", y="per_100t", title="Packaging complaints per 100 t by sealing temperature band (°C)", color_discrete_sequence=["#eb6834"], hover_data=["complaints", "months"]).update_layout(xaxis_title="average sealing temperature band (°C)", yaxis_title="complaints / 100 t", **LAYOUT), width="stretch")
    st.caption(f"Infestation rate is **{bi.per_100t.iat[-1]/bi.per_100t.iat[0]:.1f}×** higher in the wettest band than the driest; packaging complaints are **{bp.per_100t.iat[0]/bp.per_100t.iat[-1]:.1f}×** higher in the coolest-seal band than the hottest.")
    st.subheader("Who sees what? Customer, product and root-cause links")
    a1, a2 = st.columns(2)
    a1.markdown("**Customers with unusually high complaint types** (lift > 1 = more than average)"); a1.dataframe(E.lift_table(f, "customer", "category", top=10), hide_index=True, width="stretch")
    a2.markdown("**Products with unusually high root causes**"); a2.dataframe(E.lift_table(f, "product", "root_cause", top=10), hide_index=True, width="stretch")
    pr_ = pd.crosstab(f["product"], f.root_cause); top_rc = pr_.sum().nlargest(14).index
    st.plotly_chart(px.imshow(pr_[top_rc], color_continuous_scale=SEQ, aspect="auto", text_auto=True, title="Product × root cause (14 most frequent)", labels=dict(color="Complaints")).update_layout(**LAYOUT), width="stretch")

# ---------------- 6. risk ----------------
with tabs[5]:
    cur = P["cur"]; fm = cur.forecast_month.iat[0]; rm = P["rm"]
    st.subheader(f"Predicted complaint risk for {fm}")
    st.caption("Gradient-boosting model on product × category × month: recent complaint counts, seasonality, process parameters and shipment volume → probability of ≥1 complaint next month.")
    hm = cur.pivot(index="product", columns="category", values="risk_prob")
    fig = px.imshow(hm, color_continuous_scale=SEQ, zmin=0, zmax=1, aspect="auto", text_auto=".0%", labels=dict(color="Probability"), title=f"Probability of at least one complaint in {fm}")
    fig.update_layout(**LAYOUT); st.plotly_chart(fig, width="stretch")
    a, b = st.columns([3, 2])
    top = cur.sort_values("risk_prob", ascending=False).head(10).copy()
    top["risk_prob"] = (top.risk_prob * 100).round(0); top["baseline_prob"] = (top.baseline_prob * 100).round(0)
    a.markdown("**Top-10 risk combinations**"); a.dataframe(top[["product", "category", "risk_prob", "baseline_prob", "uplift", "band"]].rename(columns={"risk_prob": "risk %", "baseline_prob": "usual %", "uplift": "vs usual (x)"}), hide_index=True, width="stretch")
    pr = P["prisk"].copy(); pr["any_complaint_prob"] = (pr.any_complaint_prob * 100).round(0); pr["top_category_prob"] = (pr.top_category_prob * 100).round(0)
    b.markdown("**Product ranking**"); b.dataframe(pr[["product", "top_category", "top_category_prob", "high_risk_categories"]].rename(columns={"top_category_prob": "top risk %", "high_risk_categories": "# high"}), hide_index=True, width="stretch")
    st.subheader("High-risk processes and production lines")
    st.caption("Process risk index = Σ over products and complaint types of P(complaint next month) × P(root cause | product, type). It shows which process failure modes are expected to drive next month's complaints; “vs usual” compares with the 24-month norm.")
    p1, p2 = st.columns([3, 2])
    pc = P["proc"].head(12).sort_values("risk_index")
    fig = px.bar(pc, x="risk_index", y="root_cause", color="band", orientation="h", title="Process failure modes ranked by predicted contribution", hover_data=["process_area", "vs_usual_x"],
                 color_discrete_map={"High": RED, "Medium": "#eda100", "Low": BLUE})
    fig.update_layout(yaxis_title="", xaxis_title="risk index", height=430, **LAYOUT); p1.plotly_chart(fig, width="stretch")
    ar = P["area"].sort_values("risk_index")
    p2.plotly_chart(px.bar(ar, x="risk_index", y="process_area", orientation="h", title="By process area", color_discrete_sequence=[BLUE], hover_data=["top_failure_mode", "max_vs_usual"]).update_layout(yaxis_title="", xaxis_title="risk index", height=430, **LAYOUT), width="stretch")
    lr = P["lines"].copy(); lr["top_risk_prob"] = (lr.top_risk_prob * 100).round(0)
    st.markdown("**Production lines: expected complaint events next month vs usual**")
    st.dataframe(lr[["line", "products", "expected_events", "usual_events", "vs_usual_x", "top_risk", "top_risk_prob", "hottest_signal", "hottest_uplift"]], hide_index=True, width="stretch")
    hp = P["proc"][P["proc"].band != "Low"]
    if len(hp): st.warning("**Processes needing attention:** " + "; ".join(f"{r.root_cause} ({r.process_area}, {r.vs_usual_x}× usual)" for _, r in hp.iterrows()))
    st.subheader("🚨 Early-warning alerts")
    ew = P["ew"]
    if len(ew):
        for _, r in ew.iterrows(): st.error(f"**{r['product']}** · {r.indicator} (signal {r.z_score}) → risk of *{r.linked_category}*.  **Action:** {r.action}")
    else: st.success("No early-warning indicators breached.")
    with st.expander("Model validation (honest numbers)"):
        st.write(f"Time-based split: trained up to {rm['test_from']}, tested on the following 36 months ({rm['n_test']:,} product-category-months).")
        st.write(f"ROC-AUC **{rm['auc']:.2f}** (0.50 = chance). Brier score **{rm['brier']:.3f}** vs {rm['brier_naive']:.3f} for a no-skill baseline - lower is better. "
                 f"Complaint text classifier: accuracy **{P['cm']['accuracy']:.1%}**, macro-F1 **{P['cm']['macro_f1']:.2f}** on {P['cm']['n_test']:,} held-out complaints. "
                 "Prototype uses synthetic data; accuracy on real ABT-CPD records will differ and should be re-validated.")

# ---------------- 7. recommendations ----------------
with tabs[6]:
    st.subheader("Preventive action advisor")
    c1, c2 = st.columns(2)
    hi = cur.sort_values("risk_prob", ascending=False).iloc[0]
    pdt = c1.selectbox("Product", sorted(df_all["product"].unique()), index=sorted(df_all["product"].unique()).index(hi["product"]))
    cat = c2.selectbox("Complaint type", sorted(df_all.category.unique()), index=sorted(df_all.category.unique()).index(hi.category))
    rc = E.likely_root_causes(df_all, pdt, cat); rec, gen = E.recommend(df_all, P["abr"], pdt, cat)
    r = cur[(cur["product"] == pdt) & (cur.category == cat)]
    if len(r): st.info(f"Predicted risk of **{cat}** for **{pdt}** in {fm}: **{r.risk_prob.iat[0]:.0%}** ({r.band.iat[0]}), {r.uplift.iat[0]}× its usual level.")
    a, b = st.columns([2, 3])
    a.plotly_chart(px.bar(rc.sort_values("likelihood_%"), x="likelihood_%", y="root_cause", orientation="h", title="Likely root causes (recent history)", color_discrete_sequence=[BLUE]).update_layout(yaxis_title="", **LAYOUT), width="stretch")
    b.markdown("**Recommended actions, ranked by historical effectiveness**"); b.dataframe(rec.sort_values(["likelihood_pct", "historical_effectiveness_pct"], ascending=False), hide_index=True, width="stretch")
    st.markdown("**Standard preventive checks before dispatch:**\n" + "\n".join(f"- {g}" for g in gen))

# ---------------- 8. log complaint ----------------
with tabs[7]:
    st.subheader("Log a new complaint - the system learns from it")
    with st.form("new"):
        c1, c2, c3 = st.columns(3)
        d = c1.date_input("Complaint date"); cu = c2.selectbox("Customer", sorted(df_all.customer.unique())); pd_ = c3.selectbox("Product", sorted(df_all["product"].unique()), key="np")
        bt = st.text_input("Batch number (optional)"); tx = st.text_area("Complaint description", placeholder="e.g. Metal fragment found in Spice Mix pouch by customer")
        go_ = st.form_submit_button("Analyse complaint")
    if go_ and tx.strip():
        cat_, conf = E.classify(P["vec"], P["clf"], [tx]); cat_, conf = cat_[0], conf[0]
        st.session_state["new"] = dict(date=pd.Timestamp(d), customer=cu, product=pd_, batch_no=bt or np.nan, complaint_text=tx.strip(), cat=cat_, conf=conf)
    n = st.session_state.get("new")
    if n:
        st.success(f"Classified as **{n['cat']}** (confidence {n['conf']:.0%})")
        rec, gen = E.recommend(df_all, P["abr"], n["product"], n["cat"]); rc = E.likely_root_causes(df_all, n["product"], n["cat"])
        x, y = st.columns(2); x.markdown("**Likely root causes**"); x.dataframe(rc, hide_index=True, width="stretch")
        y.markdown("**Suggested corrective / preventive actions**"); y.dataframe(rec[["action", "historical_effectiveness_pct", "recommendation"]].drop_duplicates("action"), hide_index=True, width="stretch")
        if st.button("Save to database and refresh models"):
            raw = pd.read_csv(E.DATA); line = df_all[df_all["product"] == n["product"]].line.iat[0]
            row = dict(complaint_id=f"CMP-{len(raw)+1:05d}", date=n["date"].date(), customer=n["customer"], product=n["product"], line=line, batch_no=n["batch_no"], complaint_text=n["complaint_text"], category=n["cat"], severity="Major")
            pd.concat([raw, pd.DataFrame([row])], ignore_index=True).to_csv(E.DATA, index=False)
            st.session_state.pop("new"); st.cache_resource.clear(); st.rerun()
