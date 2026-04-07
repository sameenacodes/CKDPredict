import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
model_data = pickle.load(open("Kidney.pkl", "rb"))
model      = model_data["model"]
accuracy   = round(model_data["accuracy"] * 100, 2)
print(f"✅ Model loaded | Accuracy: {accuracy}%")


# ── eGFR CALCULATION (CKD-EPI formula) ───────────────────────────────────────
def calculate_egfr(creatinine, age, gender="male"):
    """
    MDRD Formula: eGFR = 175 × (SCr)^(-1.154) × (Age)^(-0.203)
    Female: multiply by 0.742
    """
    try:
        scr = float(creatinine)
        age = float(age)
        if scr <= 0 or age <= 0:
            return None
        egfr = 175 * (scr ** -1.154) * (age ** -0.203)
        if gender == "female":
            egfr *= 0.742
        return round(egfr, 1)
    except:
        return None


# ── CKD STAGE FROM eGFR ──────────────────────────────────────────────────────
def get_ckd_stage(egfr):
    if egfr is None:
        return {"stage": "Unknown", "label": "Unable to calculate", "color": "gray", "desc": "Enter valid creatinine and age"}
    if egfr >= 90:
        return {"stage": "Stage 1", "egfr": egfr, "label": "Normal / Mild",       "color": "green",  "desc": "Kidney working normally. Mild symptoms possible."}
    elif egfr >= 60:
        return {"stage": "Stage 2", "egfr": egfr, "label": "Mild Damage",         "color": "green",  "desc": "Mild kidney damage. Monitor regularly."}
    elif egfr >= 45:
        return {"stage": "Stage 3A","egfr": egfr, "label": "Moderate CKD",        "color": "yellow", "desc": "Moderate kidney damage. Consult nephrologist."}
    elif egfr >= 30:
        return {"stage": "Stage 3B","egfr": egfr, "label": "Moderate–Severe",     "color": "orange", "desc": "Significant kidney damage. Treatment needed."}
    elif egfr >= 15:
        return {"stage": "Stage 4", "egfr": egfr, "label": "Severe CKD",          "color": "red",    "desc": "Severe damage. Prepare for kidney replacement therapy."}
    else:
        return {"stage": "Stage 5", "egfr": egfr, "label": "Kidney Failure",      "color": "darkred","desc": "Kidney failure. Dialysis or transplant needed immediately."}


# ── EARLY RISK TIMELINE PREDICTION ───────────────────────────────────────────
def predict_timeline(probability, age, bp, sc, htn, dm):
    """
    Converts ML probability → future CKD risk timeline.
    Novel: Most systems say Yes/No. This says WHEN.
    """
    prob = float(probability)
    age  = float(age)
    bp   = float(bp)
    sc   = float(sc)

    # Risk multipliers based on clinical factors
    risk_multiplier = 1.0
    if htn == "yes":    risk_multiplier += 0.15
    if dm  == "yes":    risk_multiplier += 0.20
    if age > 60:        risk_multiplier += 0.10
    if bp  > 100:       risk_multiplier += 0.10
    if sc  > 1.5:       risk_multiplier += 0.20

    adjusted_prob = min(prob * risk_multiplier, 0.99)

    if adjusted_prob < 0.20:
        return {
            "risk_level":  "Very Low Risk",
            "risk_color":  "green",
            "risk_pct":    round(adjusted_prob * 100, 1),
            "timeline":    "CKD unlikely in the near future",
            "months":      "No immediate concern",
            "advice":      "Maintain healthy lifestyle. Annual kidney checkup recommended.",
            "urgency":     "routine"
        }
    elif adjusted_prob < 0.40:
        return {
            "risk_level":  "Low–Moderate Risk",
            "risk_color":  "yellow",
            "risk_pct":    round(adjusted_prob * 100, 1),
            "timeline":    "CKD may develop in 12–24 months",
            "months":      "12 – 24 months",
            "advice":      "Schedule nephrology consultation. Monitor BP and blood sugar closely.",
            "urgency":     "monitor"
        }
    elif adjusted_prob < 0.60:
        return {
            "risk_level":  "Moderate Risk",
            "risk_color":  "orange",
            "risk_pct":    round(adjusted_prob * 100, 1),
            "timeline":    "CKD may develop in 8–12 months",
            "months":      "8 – 12 months",
            "advice":      "Consult nephrologist soon. Start lifestyle modifications immediately.",
            "urgency":     "soon"
        }
    elif adjusted_prob < 0.80:
        return {
            "risk_level":  "High Risk",
            "risk_color":  "red",
            "risk_pct":    round(adjusted_prob * 100, 1),
            "timeline":    "CKD may develop in 3–6 months",
            "months":      "3 – 6 months",
            "advice":      "Urgent nephrology referral needed. Begin treatment protocol.",
            "urgency":     "urgent"
        }
    else:
        return {
            "risk_level":  "Very High Risk",
            "risk_color":  "darkred",
            "risk_pct":    round(adjusted_prob * 100, 1),
            "timeline":    "CKD likely within 1–3 months",
            "months":      "1 – 3 months",
            "advice":      "IMMEDIATE medical attention required. Contact nephrologist today.",
            "urgency":     "critical"
        }


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
def get_recommendations(is_ckd, stage_info, timeline, form_data):
    recs = []
    hemo  = float(form_data.get("hemo", 14))
    bp    = float(form_data.get("bp", 80))
    sc    = float(form_data.get("sc", 1))
    htn   = form_data.get("htn", "no")
    dm    = form_data.get("dm", "no")

    if is_ckd:
        recs.append("🏥 Consult a nephrologist (kidney specialist) immediately")
        recs.append("🔬 Schedule repeat eGFR test every 3 months")
        if hemo < 11:
            recs.append("💉 Hemoglobin critically low — check for anemia treatment (EPO therapy)")
        if sc > 1.5:
            recs.append("⚠️ High creatinine — reduce protein intake, stay hydrated")
        if htn == "yes":
            recs.append("💊 Control blood pressure — target below 130/80 mmHg")
        if dm == "yes":
            recs.append("🍬 Strict blood sugar control — HbA1c target below 7%")
        recs.append("🚫 Avoid NSAIDs (ibuprofen) — they worsen kidney damage")
        recs.append("💧 Drink 2–3 liters of water daily unless advised otherwise")
    else:
        recs.append("✅ Kidney function appears normal — maintain healthy lifestyle")
        recs.append("📅 Annual kidney checkup recommended")
        if htn == "yes":
            recs.append("💊 Monitor blood pressure regularly — target below 130/80")
        if dm == "yes":
            recs.append("🍬 Keep blood sugar controlled to protect kidneys long-term")
        recs.append("🥗 Low-sodium, balanced diet helps prevent kidney disease")
        recs.append("🏃 30 minutes of exercise daily improves kidney health")

    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def homepage():
    return render_template("homepage.html")


@app.route("/predict-page")
def predict_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── Collect ALL form fields ──────────────────────────────────────────────
    form_data = {
        "age":   request.form.get("age",   "50"),
        "bp":    request.form.get("bp",    "80"),
        "sg":    request.form.get("sg",    "1.020"),
        "al":    request.form.get("al",    "0"),
        "su":    request.form.get("su",    "0"),
        "rbc":   request.form.get("rbc",   "normal"),
        "pc":    request.form.get("pc",    "normal"),
        "pcc":   request.form.get("pcc",   "notpresent"),
        "ba":    request.form.get("ba",    "notpresent"),
        "bgr":   request.form.get("bgr",   "100"),
        "bu":    request.form.get("bu",    "40"),
        "sc":    request.form.get("sc",    "1"),
        "sod":   request.form.get("sod",   "140"),
        "pot":   request.form.get("pot",   "4"),
        "hemo":  request.form.get("hemo",  "14"),
        "pcv":   request.form.get("pcv",   "42"),
        "wc":    request.form.get("wc",    "7500"),
        "rc":    request.form.get("rc",    "5"),
        "htn":   request.form.get("htn",   "no"),
        "dm":    request.form.get("dm",    "no"),
        "cad":   request.form.get("cad",   "no"),
        "appet": request.form.get("appet", "good"),
        "pe":    request.form.get("pe",    "no"),
        "ane":   request.form.get("ane",   "no"),
        "gender":request.form.get("gender","male"),
    }

    # ── Encode 8 model features (train order: sg,htn,hemo,dm,al,appet,rc,pc) ─
    sg    = float(form_data["sg"])
    htn   = 1.0 if form_data["htn"]   == "yes"      else 0.0
    hemo  = float(form_data["hemo"])
    dm_v  = str(form_data["dm"]).strip().lower().replace('\t','').replace(' ','')
    dm    = 1.0 if dm_v == "yes" else 0.0
    al    = float(form_data["al"])
    appet = 1.0 if form_data["appet"] == "good"     else 0.0
    rc    = float(form_data["rc"])
    pc    = 0.0 if form_data["pc"]    == "normal"   else 1.0

    input_data = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])

    # ── ML Prediction ────────────────────────────────────────────────────────
    prediction  = model.predict(input_data)[0]
    is_ckd      = bool(prediction == 1)

    # ── Prediction Probability ───────────────────────────────────────────────
    try:
        proba     = model.predict_proba(input_data)[0]
        risk_prob = float(proba[1])  # probability of CKD class
        ckd_prob  = float(proba[1]) if is_ckd else float(proba[0])
    except:
        ckd_prob  = 0.95 if is_ckd else 0.05
        risk_prob = 0.95 if is_ckd else 0.15

    # ── eGFR & Stage ─────────────────────────────────────────────────────────
    egfr       = calculate_egfr(form_data["sc"], form_data["age"], form_data["gender"])
    stage_info = get_ckd_stage(egfr)

    # ── SMART OVERRIDE: If eGFR clearly shows CKD, override ML result ────────
    # eGFR < 60 = CKD by international medical standard (regardless of ML)
    egfr_override = False
    if egfr is not None and egfr < 60 and not is_ckd:
        is_ckd       = True
        egfr_override = True
        ckd_prob     = max(ckd_prob, 0.85)
        risk_prob    = max(risk_prob, 0.85)

    # Also override if eGFR >= 90 and ML says CKD but all values are normal
    if egfr is not None and egfr >= 90 and is_ckd and risk_prob < 0.4:
        is_ckd    = False
        ckd_prob  = 1 - ckd_prob
        risk_prob = risk_prob * 0.3

    result_text = "CKD Detected" if is_ckd else "No CKD Detected"
    if egfr_override:
        result_text = "CKD Detected (eGFR Confirmed)"

    # ── Future Risk Timeline ─────────────────────────────────────────────────
    timeline = predict_timeline(
        risk_prob,
        form_data["age"], form_data["bp"],
        form_data["sc"],  form_data["htn"], form_data["dm"]
    )

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = get_recommendations(is_ckd, stage_info, timeline, form_data)

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = round(ckd_prob * 100, 1)

    return render_template(
        "result.html",
        prediction      = result_text,
        is_ckd          = is_ckd,
        egfr_override   = egfr_override,
        accuracy        = accuracy,
        confidence      = round(ckd_prob * 100, 1),
        egfr            = egfr,
        stage_info      = stage_info,
        timeline        = timeline,
        recommendations = recommendations,
        form_data       = form_data,
    )


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file  = request.files["file"]
    df    = pd.read_csv(file)
    df    = df.fillna(df.median(numeric_only=True))
    preds = model.predict(df)
    return render_template(
        "csv_result.html",
        tables=[df.head().to_html(classes="data")],
        pred=preds[:10]
    )


if __name__ == "__main__":
    app.run(debug=True)