"""Generate a realistic SYNTHETIC 19-year complaint history for the ABT-CPD prototype.
Replace data/complaints.csv + data/monthly.csv with real ABT-CPD exports to run on real data."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(42)

START, END = "2007-11-01", "2026-10-01"          # 19 years
months = pd.date_range(START, END, freq="MS")

PRODUCTS = {  # product -> (line, base tonnes/month)
    "Rice Flour": ("L1", 900), "Wheat Flour": ("L1", 1300), "Besan": ("L2", 700),
    "Spice Mix": ("L3", 500), "Ready Mix": ("L3", 650), "Millet Snack": ("L4", 450),
    "Edible Oil Pack": ("L2", 800), "Semolina": ("L1", 600)}
CATS = ["Product Quality Deviation", "Physical Defect", "Packaging Issue", "Contamination / Foreign Matter",
        "Infestation", "Sensory Characteristics", "Weight / Quantity", "Process-Related Defect", "Other"]
CUSTOMERS = ["Retail Chain A", "Retail Chain B", "Distributor North", "Distributor South", "Export - Gulf",
             "Export - SE Asia", "Institutional Caterers", "E-commerce Partner", "Modern Trade X",
             "Wholesale Market", "Govt. Tender", "Private Label Client"]

# base complaint rate per tonne per category (relative)
CAT_RATE = {"Product Quality Deviation": .9, "Physical Defect": .7, "Packaging Issue": 1.0,
            "Contamination / Foreign Matter": .8, "Infestation": 1.1, "Sensory Characteristics": .6,
            "Weight / Quantity": .5, "Process-Related Defect": .5, "Other": .3}
# product affinity multipliers
AFF = {("Rice Flour","Infestation"):2.2, ("Wheat Flour","Infestation"):2.0, ("Semolina","Infestation"):1.8,
       ("Besan","Contamination / Foreign Matter"):1.8, ("Spice Mix","Contamination / Foreign Matter"):2.3,
       ("Spice Mix","Sensory Characteristics"):1.8, ("Edible Oil Pack","Packaging Issue"):2.4,
       ("Edible Oil Pack","Sensory Characteristics"):1.8, ("Millet Snack","Packaging Issue"):1.8,
       ("Millet Snack","Weight / Quantity"):2.0, ("Ready Mix","Process-Related Defect"):2.5,
       ("Ready Mix","Product Quality Deviation"):1.6, ("Besan","Physical Defect"):1.8,
       ("Wheat Flour","Product Quality Deviation"):1.5}
SEASON = {"Infestation": lambda m: 1+1.2*np.exp(-((m-8)**2)/6), "Sensory Characteristics": lambda m: 1+.6*np.exp(-((m-7)**2)/8),
          "Physical Defect": lambda m: 1+.5*np.exp(-((m-8)**2)/8)}
TREND_END = {"Infestation": .75, "Packaging Issue": .5, "Contamination / Foreign Matter": .45}  # improve less/more

ROOT = {
 "Infestation": ["High grain moisture at storage", "Warehouse pest-control lapse", "Pack seal breach allowing entry", "Long dwell time in distribution", "Infested raw material received"],
 "Contamination / Foreign Matter": ["Sieve / magnet failure", "Raw-material cleaning inadequate", "Maintenance debris (metal)", "Personnel hygiene lapse", "Packing-material contamination"],
 "Packaging Issue": ["Low sealing temperature", "Packaging film supplier variation", "Coding / printing error", "Carton handling damage in transit", "Machine jaw wear"],
 "Product Quality Deviation": ["Raw-material variability", "Incorrect blend ratio", "Process parameter drift", "Lab testing gap"],
 "Physical Defect": ["Moisture caking", "Milling roll wear", "Improper cooling before pack", "Conveying breakage"],
 "Sensory Characteristics": ["Rancidity due to storage", "Over / under roasting", "Raw-material odour carry-over", "Packaging barrier failure"],
 "Weight / Quantity": ["Filler calibration drift", "Check-weigher bypass", "Moisture loss in storage", "Operator set-up error"],
 "Process-Related Defect": ["Mixing non-uniformity", "Wrong grade / label dispatched", "Changeover cleaning lapse", "Sieving lapse"],
 "Other": ["Documentation / invoicing error", "Delivery delay", "Customer service gap"]}
# corrective action -> true effectiveness probability
ACTION = {
 "Install / upgrade sieve & magnetic separator": .88, "Operator retraining only": .30, "Supplier audit + incoming inspection tightened": .80,
 "Moisture control at drying + storage SOP": .85, "Fumigation + warehouse pest-control schedule": .78, "Replace customer product only": .15,
 "Sealing temperature / dwell time standardised": .86, "Preventive maintenance schedule revised": .75, "Verbal warning to operators": .20,
 "Inline check-weigher with auto-reject": .90, "Change-over cleaning checklist + verification": .82, "Process parameter alarm limits added": .84,
 "Packaging film specification re-qualified": .80, "FIFO + dwell-time limits in distribution": .72, "Revalidation of recipe / blend control": .78,
 "Apology letter to customer": .10, "Process documentation revised": .55}
ROOT_ACTIONS = {
 "High grain moisture at storage": ["Moisture control at drying + storage SOP", "Replace customer product only"],
 "Warehouse pest-control lapse": ["Fumigation + warehouse pest-control schedule", "Verbal warning to operators"],
 "Pack seal breach allowing entry": ["Sealing temperature / dwell time standardised", "Packaging film specification re-qualified"],
 "Long dwell time in distribution": ["FIFO + dwell-time limits in distribution", "Replace customer product only"],
 "Infested raw material received": ["Supplier audit + incoming inspection tightened", "Fumigation + warehouse pest-control schedule"],
 "Sieve / magnet failure": ["Install / upgrade sieve & magnetic separator", "Operator retraining only"],
 "Raw-material cleaning inadequate": ["Supplier audit + incoming inspection tightened", "Operator retraining only"],
 "Maintenance debris (metal)": ["Preventive maintenance schedule revised", "Install / upgrade sieve & magnetic separator", "Verbal warning to operators"],
 "Personnel hygiene lapse": ["Operator retraining only", "Process documentation revised"],
 "Packing-material contamination": ["Supplier audit + incoming inspection tightened", "Packaging film specification re-qualified"],
 "Low sealing temperature": ["Sealing temperature / dwell time standardised", "Process parameter alarm limits added", "Operator retraining only"],
 "Packaging film supplier variation": ["Packaging film specification re-qualified", "Supplier audit + incoming inspection tightened"],
 "Coding / printing error": ["Process documentation revised", "Operator retraining only"],
 "Carton handling damage in transit": ["FIFO + dwell-time limits in distribution", "Apology letter to customer"],
 "Machine jaw wear": ["Preventive maintenance schedule revised", "Sealing temperature / dwell time standardised"],
 "Raw-material variability": ["Supplier audit + incoming inspection tightened", "Revalidation of recipe / blend control"],
 "Incorrect blend ratio": ["Revalidation of recipe / blend control", "Operator retraining only"],
 "Process parameter drift": ["Process parameter alarm limits added", "Preventive maintenance schedule revised"],
 "Lab testing gap": ["Process documentation revised", "Supplier audit + incoming inspection tightened"],
 "Moisture caking": ["Moisture control at drying + storage SOP", "Replace customer product only"],
 "Milling roll wear": ["Preventive maintenance schedule revised", "Verbal warning to operators"],
 "Improper cooling before pack": ["Process parameter alarm limits added", "Process documentation revised"],
 "Conveying breakage": ["Preventive maintenance schedule revised", "Process documentation revised"],
 "Rancidity due to storage": ["FIFO + dwell-time limits in distribution", "Replace customer product only"],
 "Over / under roasting": ["Process parameter alarm limits added", "Operator retraining only"],
 "Raw-material odour carry-over": ["Supplier audit + incoming inspection tightened", "Changeover"],
 "Packaging barrier failure": ["Packaging film specification re-qualified", "Supplier audit + incoming inspection tightened"],
 "Filler calibration drift": ["Inline check-weigher with auto-reject", "Preventive maintenance schedule revised"],
 "Check-weigher bypass": ["Inline check-weigher with auto-reject", "Verbal warning to operators"],
 "Moisture loss in storage": ["Moisture control at drying + storage SOP", "Process documentation revised"],
 "Operator set-up error": ["Inline check-weigher with auto-reject", "Operator retraining only"],
 "Mixing non-uniformity": ["Revalidation of recipe / blend control", "Process parameter alarm limits added"],
 "Wrong grade / label dispatched": ["Change-over cleaning checklist + verification", "Operator retraining only"],
 "Changeover cleaning lapse": ["Change-over cleaning checklist + verification", "Operator retraining only"],
 "Sieving lapse": ["Install / upgrade sieve & magnetic separator", "Operator retraining only"],
 "Documentation / invoicing error": ["Process documentation revised", "Apology letter to customer"],
 "Delivery delay": ["FIFO + dwell-time limits in distribution", "Apology letter to customer"],
 "Customer service gap": ["Process documentation revised", "Apology letter to customer"]}
ROOT_ACTIONS["Raw-material odour carry-over"][1] = "Revalidation of recipe / blend control"

TEXT = {
 "Infestation": ["Live insects found in {p} pack", "Weevils observed inside sealed {p} pouch", "Larvae noticed in {p} packet close to expiry", "Webbing and insect activity seen in {p}", "Customer reports beetles in {p} bag"],
 "Contamination / Foreign Matter": ["Plastic piece found in {p}", "Metal fragment detected in {p} by consumer", "Hair strand found inside {p} pack", "Stones and grit found in {p}", "Wood splinter observed in {p}", "Foreign matter reported in {p} sample"],
 "Packaging Issue": ["Seal leakage in {p} pouch", "Pouch burst during transit of {p}", "Print smudged and batch code unreadable on {p}", "Carton damaged and pouches loose, {p}", "Weak seal opened easily in {p} pack"],
 "Product Quality Deviation": ["Product failed spec, values out of range for {p}", "Cooking quality poor, not matching standard {p}", "Texture inconsistent versus approved sample {p}", "Out of specification in customer lab test, {p}"],
 "Physical Defect": ["Lumps and clumping in {p}", "Broken and deformed pieces in {p}", "Particle size inconsistent in {p}", "Moisture caking observed in {p}"],
 "Sensory Characteristics": ["Off odour reported in {p}", "Rancid taste in {p}", "Colour darker than usual in {p}", "Stale flavour noticed in {p}", "Bitter after-taste in {p}"],
 "Weight / Quantity": ["Net weight less than declared in {p} pack", "Short quantity, pack underfilled {p}", "Pack weight variation noticed in {p} lot", "Underweight pouches found, {p}"],
 "Process-Related Defect": ["Under-roasted product received, {p}", "Mixing not uniform in {p} blend", "Wrong grade dispatched, labelled {p}", "Oversized particles from sieving lapse in {p}"],
 "Other": ["Delay in delivery and documentation issue for {p}", "Customer service response not satisfactory regarding {p}", "Invoice mismatch with dispatched {p}"]}
PREFIX = ["", "", "", "Customer complaint: ", "Escalated by customer - ", "Retailer feedback: "]
SUFFIX = ["", "", "", ". Photo attached", ". Sample returned", ". Repeat issue as per customer", ". Batch retained for analysis"]
MIXED = [("Seal leakage allowed insect entry in {p}", "Packaging Issue", "Infestation"),
         ("Insects found; pack seal appears weak in {p}", "Infestation", "Packaging Issue"),
         ("Caked lumps with musty odour in {p}", "Physical Defect", "Sensory Characteristics"),
         ("Pack underfilled and torn in {p}", "Weight / Quantity", "Packaging Issue")]

# ---- monthly process + volume data
rows = []
for p, (line, base) in PRODUCTS.items():
    for i, m in enumerate(months):
        growth = 1 + 0.025 * (i / 12)
        vol = base * growth * (1 + .1*np.sin(2*np.pi*(m.month-3)/12)) * rng.normal(1, .05)
        moist = 11.0 + 0.55*np.exp(-((m.month-8)**2)/6) + rng.normal(0, .25) - 0.0075*(i/12)*10/2
        seal = 146 + rng.normal(0, 1.5)
        if line == "L2" and pd.Timestamp("2016-05-01") <= m <= pd.Timestamp("2017-03-01"): seal -= 5   # historical film issue
        if p == "Wheat Flour" and m >= pd.Timestamp("2026-08-01"): moist += 0.9 + 0.2*(m.month-8)   # recent early-warning signal
        rows.append(dict(product=p, line=line, month=m, volume_tonnes=round(vol, 1), avg_moisture=round(moist, 2), avg_seal_temp=round(seal, 1)))
monthly = pd.DataFrame(rows)

# ---- complaints
recs = []
n_months = len(months)
for _, r in monthly.iterrows():
    i = months.get_loc(r.month); frac = i / (n_months-1)
    for cat in CATS:
        rate = CAT_RATE[cat] * AFF.get((r["product"], cat), 1.0) * 0.0007
        trend = 1 - (1 - TREND_END.get(cat, .55)) * frac
        rate *= trend * SEASON.get(cat, lambda m: 1)(r.month.month)
        if cat == "Infestation": rate *= np.exp(0.75*(r.avg_moisture-11.2))
        if cat == "Packaging Issue": rate *= np.exp(-0.18*(r.avg_seal_temp-146))
        if cat == "Contamination / Foreign Matter" and r.line == "L3" and r.month >= pd.Timestamp("2026-06-01"): rate *= 6.0  # recent signal
        if cat == "Weight / Quantity" and r["product"] == "Millet Snack" and r.month.year in (2013, 2014): rate *= 2.0
        n = rng.poisson(rate * r.volume_tonnes)
        for _ in range(n):
            day = int(rng.integers(1, 28)); d = r.month + pd.Timedelta(days=day-1)
            true_cat = cat; tmpl = None
            if rng.random() < .06:
                mt = MIXED[int(rng.integers(len(MIXED)))]; tmpl = mt[0]; true_cat = mt[1] if rng.random() < .55 else mt[2]
            else:
                tmpl = TEXT[cat][int(rng.integers(len(TEXT[cat])))]
            text = PREFIX[int(rng.integers(len(PREFIX)))] + tmpl.format(p=r["product"]) + SUFFIX[int(rng.integers(len(SUFFIX)))]
            true_cat = cat if tmpl in TEXT[cat] else true_cat
            # customer: affinity for certain customers
            w = np.ones(len(CUSTOMERS))
            if cat == "Contamination / Foreign Matter": w[[4, 5]] *= 2.5
            if cat == "Packaging Issue": w[[0, 7]] *= 2.2
            if cat == "Weight / Quantity": w[[10, 6]] *= 2.5
            if cat == "Infestation": w[[2, 3, 9]] *= 2.2
            cust = rng.choice(CUSTOMERS, p=w/w.sum())
            # root cause (weighted by recent/process conditions)
            rc_list = ROOT[true_cat]; rw = np.ones(len(rc_list))
            if true_cat == "Infestation" and r.avg_moisture > 11.5: rw[0] *= 3
            if true_cat == "Packaging Issue" and r.avg_seal_temp < 143: rw[0] *= 4
            if true_cat == "Contamination / Foreign Matter" and r.line == "L3" and r.month >= pd.Timestamp("2026-06-01"): rw[2] *= 4
            rw = rw * np.linspace(1.4, .7, len(rw))
            rc = rng.choice(rc_list, p=rw/rw.sum())
            act = rng.choice(ROOT_ACTIONS[rc], p=[.6] + [.4/(len(ROOT_ACTIONS[rc])-1)]*(len(ROOT_ACTIONS[rc])-1)) if len(ROOT_ACTIONS[rc]) > 1 else ROOT_ACTIONS[rc][0]
            eff = bool(rng.random() < ACTION[act])
            sev = "Critical" if (true_cat in ("Contamination / Foreign Matter",) and rng.random() < .5) or (true_cat == "Infestation" and rng.random() < .3) else ("Major" if rng.random() < .45 else "Minor")
            rec = dict(date=d, customer=cust, product=r["product"], line=r.line, batch_no=f"{r.line}-{d:%y%m}-{int(rng.integers(1,60)):03d}",
                       complaint_text=text, category=true_cat, severity=sev,
                       batch_moisture=round(r.avg_moisture + rng.normal(0, .3), 2), batch_seal_temp=round(r.avg_seal_temp + rng.normal(0, 1.2), 1),
                       root_cause=rc, corrective_action=act, action_effective=eff,
                       closure_days=int(np.clip(rng.lognormal(2.7 - .4*frac, .5), 2, 90)))
            # realistic gaps in older records
            if d.year < 2013:
                if rng.random() < .35: rec["root_cause"] = np.nan; rec["corrective_action"] = np.nan; rec["action_effective"] = np.nan
                if rng.random() < .6: rec["batch_no"] = np.nan; rec["batch_moisture"] = np.nan; rec["batch_seal_temp"] = np.nan
            elif rng.random() < .08:
                rec["root_cause"] = np.nan; rec["corrective_action"] = np.nan; rec["action_effective"] = np.nan
            if rng.random() < .18: rec["category"] = np.nan   # unlabeled -> AI classification fills it
            recs.append(rec)
df = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
df.insert(0, "complaint_id", [f"CMP-{i+1:05d}" for i in range(len(df))])
os.makedirs("data", exist_ok=True)
df.to_csv("data/complaints.csv", index=False); monthly.to_csv("data/monthly.csv", index=False)
print(len(df), "complaints;", df.date.min().date(), "->", df.date.max().date())
print(df.category.value_counts(dropna=False)); print(df.groupby(df.date.dt.year).size().to_string())
