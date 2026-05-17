"""
Credit Risk Scorecard & Explainability Dashboard
Vaidik Sharma | github.com/Vaidik6920
Live scoring | Model Insights | Batch Scoring | Risk Analytics
"""

import time, json, warnings
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from io import BytesIO

warnings.filterwarnings("ignore")

API_BASE = "https://credit-risk-intelligence-xv4z.onrender.com"

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background:#0f172a; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }

.risk-card { border-radius:14px; padding:24px 28px; text-align:center; margin-bottom:12px; }
.risk-low    { background:linear-gradient(135deg,#dcfce7,#bbf7d0); border:2px solid #16a34a; }
.risk-medium { background:linear-gradient(135deg,#fef9c3,#fde68a); border:2px solid #d97706; }
.risk-high   { background:linear-gradient(135deg,#fee2e2,#fecaca); border:2px solid #dc2626; }
.risk-vhigh  { background:linear-gradient(135deg,#f3e8ff,#e9d5ff); border:2px solid #7c3aed; }
.score-number { font-size:60px; font-weight:800; line-height:1; }
.score-label  { font-size:20px; font-weight:700; margin-top:6px; }
.score-action { font-size:14px; margin-top:8px; opacity:0.85; }
.prob-text    { font-size:14px; margin-top:10px; }

.ratio-card {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:12px 16px; margin:6px 0;
    display:flex; justify-content:space-between; align-items:center;
}
.ratio-label { font-size:12px; color:#64748b; }
.ratio-val   { font-size:18px; font-weight:700; }
.ratio-good  { color:#16a34a; } .ratio-warn { color:#d97706; } .ratio-bad { color:#dc2626; }

.tip-card {
    border-left:4px solid; border-radius:8px;
    padding:12px 16px; margin:8px 0; background:#f8fafc;
}
.tip-high   { border-color:#dc2626; }
.tip-medium { border-color:#d97706; }
.tip-low    { border-color:#16a34a; }
.tip-title  { font-size:14px; font-weight:700; margin-bottom:4px; }
.tip-body   { font-size:13px; color:#475569; line-height:1.5; }

.score-input-box {
    background:#f0f9ff; border:1.5px solid #bae6fd; border-radius:10px;
    padding:14px 18px; margin:6px 0;
}
.score-input-label { font-size:12px; color:#0369a1; font-weight:600; margin-bottom:4px; }
.score-range { font-size:11px; color:#64748b; margin-top:4px; }

.field-hint { font-size:11px; color:#94a3b8; margin-top:2px; }

.section-header {
    font-size:14px; font-weight:700; color:#1e40af;
    padding:8px 0 4px 0; border-bottom:2px solid #e2e8f0; margin-bottom:12px;
}

.improve-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:16px; margin:8px 0;
}
.improve-title { font-size:14px; font-weight:700; color:#1e293b; margin-bottom:6px; }
.improve-body  { font-size:13px; color:#475569; line-height:1.6; }
.improve-impact{ font-size:12px; font-weight:600; margin-top:8px; padding:4px 8px;
                 border-radius:6px; display:inline-block; }
.impact-high   { background:#dcfce7; color:#16a34a; }
.impact-medium { background:#fef9c3; color:#d97706; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_model_info():
    try:
        r = requests.get(f"{API_BASE}/model/info", timeout=30)
        return r.json() if r.status_code == 200 else {}
    except: return {}

@st.cache_data(ttl=60)
def get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=30)
        return r.json() if r.status_code == 200 else {}
    except: return {}

def predict_single(payload, explain=True):
    payload["explain"] = explain
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)
        if r.status_code == 200: return r.json()
        elif r.status_code == 503: return {"error":"cold_start"}
        else: return {"error": r.text}
    except requests.exceptions.Timeout: return {"error":"timeout"}
    except Exception as e: return {"error": str(e)}

def predict_batch(records):
    try:
        r = requests.post(f"{API_BASE}/predict/batch",
                          json={"applications": records, "return_shap": False}, timeout=120)
        return r.json() if r.status_code == 200 else None
    except: return None

def risk_css(label):
    return {"Low Risk":"risk-low","Medium Risk":"risk-medium",
            "High Risk":"risk-high","Very High Risk":"risk-vhigh"}.get(label,"risk-medium")

def risk_emoji(label):
    return {"Low Risk":"🟢","Medium Risk":"🟡","High Risk":"🔴","Very High Risk":"🟣"}.get(label,"🟡")

def ratio_color(val, good, warn):
    if val <= good: return "ratio-good"
    elif val <= warn: return "ratio-warn"
    else: return "ratio-bad"

# ── IMPROVEMENT TIPS ENGINE ───────────────────────────────────────────────────
def generate_improvement_tips(payload, result):
    """Generate specific, actionable tips based on applicant data and model output."""
    tips = []
    prob = result.get("default_probability", 0.5)

    income  = payload.get("AMT_INCOME_TOTAL", 100000)
    credit  = payload.get("AMT_CREDIT", 300000)
    annuity = payload.get("AMT_ANNUITY", 20000)
    age_days= payload.get("DAYS_BIRTH", -14000)
    emp_days= payload.get("DAYS_EMPLOYED", -1000)
    ext1    = payload.get("EXT_SOURCE_1", 0.5) or 0.5
    ext2    = payload.get("EXT_SOURCE_2", 0.5) or 0.5
    ext3    = payload.get("EXT_SOURCE_3", 0.5) or 0.5

    cti  = credit / (income + 1)
    ati  = annuity / (income + 1)
    ext_avg = (ext1 + ext2 + ext3) / 3
    emp_yrs = abs(emp_days) / 365.25

    # Credit-to-income
    if cti > 5:
        tips.append({"priority":"high","title":"Reduce Loan Amount",
            "body":f"Your credit-to-income ratio is <b>{cti:.1f}x</b> — significantly above the safe threshold of 4x. "
                   f"Reducing the loan from <b>R${credit:,.0f}</b> to <b>R${income*4:,.0f}</b> would bring this ratio into the acceptable range.",
            "impact":"High Impact","css":"impact-high"})
    elif cti > 4:
        tips.append({"priority":"medium","title":"Loan Amount is on the Edge",
            "body":f"Credit-to-income ratio of <b>{cti:.1f}x</b> is above ideal (4x). "
                   f"Consider reducing loan by R${(credit - income*4):,.0f} or increasing income documentation.",
            "impact":"Medium Impact","css":"impact-medium"})

    # Annuity burden
    if ati > 0.40:
        tips.append({"priority":"high","title":"Monthly Repayment Too High",
            "body":f"Annual repayment is <b>{ati*100:.0f}%</b> of income — above the 40% safe limit. "
                   f"Extending loan tenure would reduce monthly instalments. "
                   f"Target: keep repayment under R${income*0.35:,.0f}/year.",
            "impact":"High Impact","css":"impact-high"})
    elif ati > 0.30:
        tips.append({"priority":"medium","title":"Repayment Burden is Moderate",
            "body":f"Repayment-to-income at <b>{ati*100:.0f}%</b>. Consider extending tenure to reduce monthly burden below 30% of income.",
            "impact":"Medium Impact","css":"impact-medium"})

    # External scores
    if ext_avg < 0.35:
        tips.append({"priority":"high","title":"External Credit Scores are Low",
            "body":f"Average external credit score is <b>{ext_avg:.2f}</b> (low risk threshold: 0.50+). "
                   "These scores reflect your credit bureau history. Steps to improve: "
                   "clear any outstanding dues, avoid multiple loan enquiries in 3 months, "
                   "ensure credit card bills are paid on time for 6+ months.",
            "impact":"High Impact","css":"impact-high"})
    elif ext_avg < 0.50:
        tips.append({"priority":"medium","title":"Credit Scores Have Room to Improve",
            "body":f"External score average of <b>{ext_avg:.2f}</b> is below the 0.50 benchmark. "
                   "Clearing any overdue EMIs and maintaining zero late payments for 90 days can meaningfully improve this.",
            "impact":"Medium Impact","css":"impact-medium"})

    # Employment
    if emp_yrs < 1:
        tips.append({"priority":"high","title":"Very Short Employment History",
            "body":f"Only <b>{emp_yrs:.1f} years</b> of current employment. "
                   "Lenders prefer 2+ years at the same employer as a stability signal. "
                   "If switching jobs, consider applying after a 6-month stabilisation period.",
            "impact":"High Impact","css":"impact-high"})
    elif emp_yrs < 2:
        tips.append({"priority":"medium","title":"Employment Tenure is Short",
            "body":f"<b>{emp_yrs:.1f} years</b> at current employer. "
                   "Waiting 6-12 more months would significantly strengthen the employment stability signal.",
            "impact":"Medium Impact","css":"impact-medium"})

    # Age
    age_yrs = abs(age_days) / 365.25
    if age_yrs < 25:
        tips.append({"priority":"medium","title":"Young Applicant — Limited Credit History",
            "body":f"At <b>{age_yrs:.0f} years</b>, credit history is limited. "
                   "Building a positive track record with a smaller loan first (personal loan, credit card) "
                   "can raise external scores before applying for a larger loan.",
            "impact":"Medium Impact","css":"impact-medium"})

    # Already low risk
    if prob < 0.10 and not tips:
        tips.append({"priority":"low","title":"Profile is Strong",
            "body":"Your credit profile is already in good standing. "
                   "Maintaining current employment stability and keeping credit utilisation below 30% "
                   "will preserve and improve your score over time.",
            "impact":"Low Risk","css":"impact-high"})

    return tips


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 🏦 Credit Risk Scorecard")
        st.markdown("XGBoost + LightGBM + CatBoost")
        st.markdown("**AUC-ROC: 0.7899** · 271 features"); st.markdown("---")
        page = st.radio("Navigate", [
            "🎯 Live Scoring",
            "📊 Model Insights",
            "📋 Batch Scoring",
            "📈 Risk Analytics",
        ])
        st.markdown("---")
        health = get_health()
        if health.get("status") == "ready":
            st.success("API Online ✅")
            st.caption(f"Predictions served: {health.get('total_predictions',0):,}")
        elif health.get("status") == "loading":
            st.warning("Models Loading ⏳")
        else:
            st.error("API Offline ❌")
            st.caption("Render free tier — may take 30s to wake")
        st.markdown("---")
        st.markdown("**Vaidik Sharma** · IIT KGP 2026")
        st.markdown(f"[API Docs]({API_BASE}/docs) · [GitHub](https://github.com/Vaidik6920)")
    return page


# ── PAGE 1: LIVE SCORING ──────────────────────────────────────────────────────
def page_live_scoring():
    st.title("🎯 Live Credit Default Prediction")
    st.markdown("Fill in the applicant profile below and get an instant risk score with explanations.")
    st.markdown("---")

    # ── LIVE RATIO PREVIEW (updates before submit) ────────────────────────────
    with st.expander("📐 **Live Ratio Calculator** — see risk indicators as you type", expanded=True):
        rc1, rc2, rc3 = st.columns(3)
        prev_income  = rc1.number_input("Annual Income (R$)", 10000, 10000000, 135000, 5000, key="prev_inc")
        prev_credit  = rc2.number_input("Loan Amount (R$)",   10000, 5000000,  300000, 10000, key="prev_cred")
        prev_annuity = rc3.number_input("Annual Repayment (R$)", 1000, 500000, 20000, 1000, key="prev_ann")

        cti = prev_credit / (prev_income + 1)
        ati = prev_annuity / (prev_income + 1) * 100

        m1, m2, m3, m4 = st.columns(4)

        # Credit-to-income
        cti_color = "🟢" if cti <= 4 else ("🟡" if cti <= 6 else "🔴")
        cti_label = "Safe" if cti <= 4 else ("Caution" if cti <= 6 else "High Risk")
        m1.metric("Credit-to-Income", f"{cti:.1f}x", cti_label)

        # Annuity burden
        ati_color = "🟢" if ati <= 30 else ("🟡" if ati <= 40 else "🔴")
        ati_label = "Comfortable" if ati <= 30 else ("Moderate" if ati <= 40 else "Too High")
        m2.metric("Repayment Burden", f"{ati:.0f}%", ati_label)

        # Implied tenure
        if prev_annuity > 0:
            tenure = prev_credit / prev_annuity
            m3.metric("Implied Tenure", f"{tenure:.1f} yrs", "loan years")
        else:
            m3.metric("Implied Tenure", "—", "")

        # Safe loan cap
        safe_loan = prev_income * 4
        m4.metric("Safe Loan Cap (4x income)", f"R${safe_loan:,.0f}",
                  "✅ Within" if prev_credit <= safe_loan else f"🔴 Exceeds by R${prev_credit-safe_loan:,.0f}")

    st.markdown("---")

    # ── FORM ─────────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("📝 Full Applicant Profile")

        col1, col2, col3 = st.columns(3)

        # ── FINANCIAL ────────────────────────────────────────────────────────
        with col1:
            st.markdown('<div class="section-header">💰 Financial Details</div>', unsafe_allow_html=True)
            income  = st.number_input("Annual Income (R$)", 10000, 10000000, 135000, 5000,
                                      help="Total annual income from all sources before tax")
            st.markdown('<div class="field-hint">💡 Include salary, rental, business income</div>', unsafe_allow_html=True)

            credit  = st.number_input("Loan Amount Requested (R$)", 10000, 5000000, 406597, 10000,
                                      help="Total loan principal being applied for")
            st.markdown('<div class="field-hint">💡 Keep below 4× annual income for best approval odds</div>', unsafe_allow_html=True)

            annuity = st.number_input("Annual Loan Repayment (R$)", 1000, 500000, 24700, 1000,
                                      help="Total yearly repayment amount (principal + interest)")
            st.markdown('<div class="field-hint">💡 Aim for under 30% of annual income</div>', unsafe_allow_html=True)

            goods   = st.number_input("Goods Price (R$)", 10000, 5000000, 351000, 10000,
                                      help="Price of the goods/property being financed")

        # ── PERSONAL ─────────────────────────────────────────────────────────
        with col2:
            st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
            age     = st.slider("Age (years)", 18, 70, 39,
                                help="Older applicants statistically default less")
            employed= st.slider("Years at Current Job", 0, 40, 6,
                                help="Longer tenure = more stable income signal. 2+ years is the threshold.")
            children= st.slider("Number of Children", 0, 10, 1)
            family  = st.slider("Total Family Members", 1, 10, 3,
                                help="Affects income-per-person calculation")

            st.markdown('<div class="section-header" style="margin-top:12px">🏠 Assets & Demographics</div>', unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["M","F"])
            own_car    = st.selectbox("Owns a Car", ["Y — Yes","N — No"])
            own_realty = st.selectbox("Owns Property", ["Y — Yes","N — No"])

        # ── CREDIT SCORES ─────────────────────────────────────────────────────
        with col3:
            st.markdown('<div class="section-header">📊 External Credit Scores</div>', unsafe_allow_html=True)

            st.markdown("""
            > These are **third-party bureau scores** (0.0 to 1.0).
            > They are the **most predictive features** in the model.
            > Higher = better creditworthiness.
            """)

            st.markdown("""
            | Score | Range | Meaning |
            |-------|-------|---------|
            | 0.0–0.3 | 🔴 Poor | High default risk |
            | 0.3–0.5 | 🟡 Fair | Moderate risk |
            | 0.5–0.7 | 🟢 Good | Low risk |
            | 0.7–1.0 | ✅ Excellent | Very low risk |
            """)

            ext2 = st.slider("External Score 2 (most important)", 0.0, 1.0, 0.64, 0.01,
                             help="Strongest single predictor. SHAP rank #1.")
            ext3 = st.slider("External Score 3", 0.0, 1.0, 0.31, 0.01,
                             help="SHAP rank #2. Bureau repayment history.")
            ext1 = st.slider("External Score 1", 0.0, 1.0, 0.52, 0.01,
                             help="SHAP rank #3. Third bureau source.")

            avg_ext = (ext1+ext2+ext3)/3
            color   = "#16a34a" if avg_ext >= 0.5 else ("#d97706" if avg_ext >= 0.35 else "#dc2626")
            label   = "Good" if avg_ext >= 0.5 else ("Fair" if avg_ext >= 0.35 else "Poor")
            st.markdown(f"""<div style="background:#f8fafc;border-radius:8px;padding:10px 14px;
                margin-top:8px;border:1px solid #e2e8f0;">
                <div style="font-size:12px;color:#64748b">Combined Score Average</div>
                <div style="font-size:22px;font-weight:700;color:{color}">{avg_ext:.2f} — {label}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:12px">📋 Loan Profile</div>', unsafe_allow_html=True)
            contract    = st.selectbox("Contract Type", ["Cash loans","Revolving loans"])
            income_type = st.selectbox("Income Type", ["Working","State servant",
                                                        "Commercial associate","Pensioner",
                                                        "Unemployed","Student"])
            education   = st.selectbox("Education Level", ["Higher education",
                                                             "Secondary / secondary special",
                                                             "Incomplete higher",
                                                             "Lower secondary"])

        st.markdown("---")
        submitted = st.form_submit_button("⚡  Predict Default Risk", use_container_width=True)

    # ── PREDICTION ────────────────────────────────────────────────────────────
    if submitted:
        payload = {
            "AMT_INCOME_TOTAL"   : float(income),
            "AMT_CREDIT"         : float(credit),
            "AMT_ANNUITY"        : float(annuity),
            "AMT_GOODS_PRICE"    : float(goods),
            "DAYS_BIRTH"         : -int(age * 365.25),
            "DAYS_EMPLOYED"      : -int(employed * 365.25) if employed > 0 else 0,
            "CNT_CHILDREN"       : int(children),
            "CNT_FAM_MEMBERS"    : float(family),
            "EXT_SOURCE_1"       : float(ext1),
            "EXT_SOURCE_2"       : float(ext2),
            "EXT_SOURCE_3"       : float(ext3),
            "CODE_GENDER"        : gender,
            "FLAG_OWN_CAR"       : own_car[0],
            "FLAG_OWN_REALTY"    : own_realty[0],
            "NAME_CONTRACT_TYPE" : contract,
            "NAME_INCOME_TYPE"   : income_type,
            "NAME_EDUCATION_TYPE": education,
        }

        with st.spinner("Analysing applicant profile..."):
            result = predict_single(payload, explain=True)

        if result is None or "error" in result:
            err = result.get("error","") if result else ""
            if err in ("cold_start","timeout"):
                st.warning("⏳ Server is waking up. Wait 30 seconds and click Predict again.")
            else:
                st.error(f"API error: {err}")
            return

        prob    = result["default_probability"]
        label   = result["risk_label"]
        score   = result["risk_score"]
        action  = result["recommended_action"]
        factors = result.get("top_risk_factors", [])

        st.markdown("---")
        st.subheader("📊 Assessment Results")

        # ── ROW 1: Score card + gauge + ratios ────────────────────────────────
        col1, col2, col3 = st.columns([1.1, 1, 1])

        with col1:
            css = risk_css(label)
            st.markdown(f"""<div class="risk-card {css}">
                <div class="score-number">{score}</div>
                <div class="score-label">{risk_emoji(label)} {label}</div>
                <div class="score-action">{action}</div>
                <div class="prob-text">Default probability: <b>{prob*100:.2f}%</b></div>
                <div style="font-size:11px;margin-top:4px;opacity:0.7">
                    Average applicant: 8.07% · Score 0=Riskiest, 1000=Safest
                </div>
            </div>""", unsafe_allow_html=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob*100,
                title={"text":"Default Probability %","font":{"size":13}},
                delta={"reference":8.07,"increasing":{"color":"#dc2626"},
                       "decreasing":{"color":"#16a34a"},"suffix":"pp vs avg"},
                gauge={
                    "axis":{"range":[0,100],"tickwidth":1},
                    "bar":{"color":"#1e40af"},
                    "steps":[{"range":[0,10],"color":"#dcfce7"},
                              {"range":[10,25],"color":"#fef9c3"},
                              {"range":[25,50],"color":"#fee2e2"},
                              {"range":[50,100],"color":"#f3e8ff"}],
                    "threshold":{"line":{"color":"#dc2626","width":3},
                                 "thickness":0.75,"value":8.07},
                },
            ))
            fig.update_layout(height=240, margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("**📐 Key Financial Ratios**")
            cti = credit / (income+1)
            ati = annuity / (income+1) * 100
            ext_avg2 = (ext1+ext2+ext3)/3

            for lbl, val, g, w, fmt in [
                ("Credit-to-Income", cti, 4, 6, f"{cti:.1f}x"),
                ("Repayment Burden", ati/100, 0.30, 0.40, f"{ati:.0f}%"),
                ("Avg Credit Score", ext_avg2, 0.5, 0.35, f"{ext_avg2:.2f}"),
                ("Employment Yrs",   employed, 2, 1, f"{employed} yrs"),
            ]:
                if lbl == "Avg Credit Score":
                    ok = val >= g; warn = val >= w
                    cls = "ratio-good" if ok else ("ratio-warn" if warn else "ratio-bad")
                    icon = "✅" if ok else ("⚠️" if warn else "🔴")
                elif lbl == "Employment Yrs":
                    ok = val >= g; warn = val >= w
                    cls = "ratio-good" if ok else ("ratio-warn" if warn else "ratio-bad")
                    icon = "✅" if ok else ("⚠️" if warn else "🔴")
                else:
                    bad = val > g; warn_v = val > w if lbl == "Credit-to-Income" else val > 0.30
                    cls = "ratio-bad" if bad else ("ratio-warn" if val > w else "ratio-good")
                    icon = "🔴" if bad else ("⚠️" if warn_v else "✅")
                st.markdown(f"""<div class="ratio-card">
                    <div><div class="ratio-label">{lbl}</div>
                    <div class="ratio-val {cls}">{icon} {fmt}</div></div>
                </div>""", unsafe_allow_html=True)

        # ── ROW 2: SHAP + What this means ────────────────────────────────────
        if factors:
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔍 What Drove This Score")
                st.caption("Positive SHAP = increases default risk · Negative = reduces risk")
                fig = go.Figure(go.Bar(
                    x=[f["shap_value"] for f in factors],
                    y=[f["feature"] for f in factors],
                    orientation="h",
                    marker_color=["#dc2626" if f["direction"]=="increases_risk"
                                  else "#16a34a" for f in factors],
                    text=[f"{f['shap_value']:+.4f}" for f in factors],
                    textposition="outside",
                ))
                fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
                fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=60),
                                  xaxis_title="SHAP Contribution")
                st.plotly_chart(fig, use_container_width=True)

                for f in factors:
                    is_risk = f["direction"] == "increases_risk"
                    css = "tip-high" if is_risk else "tip-low"
                    arrow = "🔺 Increases risk" if is_risk else "🔻 Reduces risk"
                    st.markdown(f"""<div class="tip-card {css}">
                        <div class="tip-title">{f['feature']} — {arrow}</div>
                        <div class="tip-body">{f['description']}</div>
                    </div>""", unsafe_allow_html=True)

            with col2:
                st.subheader("💡 What This Means in Plain English")

                # Plain english verdict
                if prob < 0.10:
                    st.success(f"""
                    **This applicant is a low credit risk.**

                    The model estimates only a **{prob*100:.1f}% chance of default** — well below the
                    population average of 8.07%. The strong external credit scores
                    (avg {ext_avg2:.2f}) and manageable credit burden ({credit/income:.1f}x income)
                    are the main positive signals.

                    **Recommended: Approve** — standard terms apply.
                    """)
                elif prob < 0.25:
                    st.warning(f"""
                    **This applicant carries moderate risk.**

                    Default probability of **{prob*100:.1f}%** is above average. The model
                    flags {sum(1 for f in factors if f['direction']=='increases_risk')} risk
                    factors. Consider approving with a reduced loan amount or requiring
                    additional collateral.

                    **Recommended: Approve with conditions.**
                    """)
                elif prob < 0.50:
                    st.error(f"""
                    **This applicant is high risk.**

                    A **{prob*100:.1f}% default probability** is {prob/0.0807:.1f}x the population
                    average. Key concerns: {
                    ', '.join([f["feature"] for f in factors
                               if f["direction"]=="increases_risk"][:3])}.

                    **Recommended: Manual review or decline.**
                    """)
                else:
                    st.error(f"""
                    **Very high default risk — decline recommended.**

                    At **{prob*100:.1f}%** probability, this application carries
                    {prob/0.0807:.1f}x the average default risk. Multiple critical
                    risk factors are present.
                    """)

                # Comparison to population
                st.markdown("**📊 Where this applicant sits**")
                buckets = ["Very Low\n(<5%)", "Low\n(5-10%)", "Medium\n(10-25%)",
                           "High\n(25-50%)", "Very High\n(>50%)"]
                thresholds = [5, 10, 25, 50, 100]
                pct = prob * 100
                colors = ["#16a34a","#4ade80","#d97706","#dc2626","#7c3aed"]
                idx = next((i for i,t in enumerate(thresholds) if pct <= t), 4)
                bar_vals = [0]*5; bar_vals[idx] = 1
                fig = go.Figure(go.Bar(x=buckets, y=[1]*5,
                                       marker_color=["rgba(37,99,235,0.15)"]*5,
                                       showlegend=False))
                fig.add_trace(go.Bar(x=[buckets[idx]], y=[1],
                                     marker_color=colors[idx], showlegend=False,
                                     text=[f"This applicant\n{pct:.1f}%"],
                                     textposition="inside"))
                fig.update_layout(barmode="overlay", height=160,
                                  yaxis=dict(showticklabels=False, showgrid=False),
                                  margin=dict(t=10,b=10,l=10,r=10),
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        # ── ROW 3: HOW TO IMPROVE ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🚀 How to Improve This Application")

        tips = generate_improvement_tips(payload, result)

        if not tips:
            st.success("✅ This profile is already strong. No major improvements needed.")
        else:
            high   = [t for t in tips if t["priority"]=="high"]
            medium = [t for t in tips if t["priority"]=="medium"]
            low    = [t for t in tips if t["priority"]=="low"]

            if high:
                st.markdown("**🔴 Address these first (High Impact)**")
                cols = st.columns(min(len(high), 2))
                for col, tip in zip(cols, high):
                    col.markdown(f"""<div class="improve-card">
                        <div class="improve-title">{tip['title']}</div>
                        <div class="improve-body">{tip['body']}</div>
                        <span class="improve-impact {tip['css']}">{tip['impact']}</span>
                    </div>""", unsafe_allow_html=True)

            if medium:
                st.markdown("**🟡 Secondary improvements (Medium Impact)**")
                cols = st.columns(min(len(medium), 2))
                for col, tip in zip(cols, medium):
                    col.markdown(f"""<div class="improve-card">
                        <div class="improve-title">{tip['title']}</div>
                        <div class="improve-body">{tip['body']}</div>
                        <span class="improve-impact {tip['css']}">{tip['impact']}</span>
                    </div>""", unsafe_allow_html=True)

            if low:
                for tip in low:
                    st.markdown(f"""<div class="improve-card">
                        <div class="improve-title">✅ {tip['title']}</div>
                        <div class="improve-body">{tip['body']}</div>
                    </div>""", unsafe_allow_html=True)

            # Quick win summary
            if high or medium:
                st.info(f"""
                💡 **Quick summary:** If the top recommendations above are addressed,
                this application's default probability could reduce by an estimated
                **{min(len(high)*8 + len(medium)*4, 30)}–{min(len(high)*15 + len(medium)*8, 50)}%**
                relative to the current score.
                """)


# ── PAGE 2: MODEL INSIGHTS ────────────────────────────────────────────────────
def page_model_insights():
    st.title("📊 Model Explainability & Performance")
    info = get_model_info()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl in [(c1,f"{info.get('training_auc',0.7899):.4f}","AUC-ROC"),
                         (c2,"+12 pp","vs LR Baseline"),
                         (c3,f"{info.get('feature_count',271)}","Features"),
                         (c4,"XGB+LGB+CAT","Ensemble"),
                         (c5,f"{info.get('optimal_threshold',0.21):.2f}","Threshold")]:
        col.markdown(f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:16px;text-align:center;">
            <div style="font-size:26px;font-weight:800;color:#1e40af">{val}</div>
            <div style="font-size:12px;color:#64748b;margin-top:4px">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 SHAP Features")
        feats = info.get("top_features", ["EXT_SOURCE_2","EXT_SOURCE_3","EXT_SOURCE_1",
            "inst_late_rate","DAYS_BIRTH","bureau_overdue_sum","CREDIT_TO_INCOME",
            "DAYS_EMPLOYED","cc_utilization_mean","prev_approval_rate"])
        imp = [1.0,0.85,0.72,0.61,0.55,0.48,0.43,0.39,0.34,0.29]
        df_fi = pd.DataFrame({"feature":feats[:10],"importance":imp[:len(feats)]})
        fig = px.bar(df_fi.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color="importance", color_continuous_scale="Blues",
                     labels={"importance":"Mean |SHAP|","feature":""})
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=10,r=20,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("AUC-ROC Progression")
        models = ["LR Baseline","XGBoost","LightGBM","Ensemble ⭐"]
        aucs   = [0.730,0.785,0.789,0.7899]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=models, y=aucs,
                             marker_color=["#94a3b8","#2563EB","#7C3AED","#16a34a"],
                             text=[f"{a:.4f}" for a in aucs], textposition="outside"))
        fig.add_hline(y=0.79, line_dash="dash", line_color="#dc2626",
                      annotation_text="0.79 target", annotation_position="right")
        fig.update_layout(height=240, yaxis=dict(range=[0.68,0.81]),
                          margin=dict(t=10,b=10,l=10,r=60))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ensemble Weights")
        weights = info.get("ensemble_weights",{"xgboost":0.55,"lightgbm":0.45,"catboost":0.0})
        fig = go.Figure(go.Pie(labels=list(weights.keys()),values=list(weights.values()),
                               hole=0.5, marker=dict(colors=["#2563EB","#DC2626","#16A34A"]),
                               textinfo="label+percent"))
        fig.update_layout(height=200, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Score Distribution")
    np.random.seed(42)
    good = np.random.beta(8,2,9193)*1000; bad = np.random.beta(2,6,807)*1000
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=good,name="Non-Default (92%)",nbinsx=50,
                               opacity=0.6,marker_color="#2563EB"))
    fig.add_trace(go.Histogram(x=bad,name="Default (8%)",nbinsx=50,
                               opacity=0.6,marker_color="#DC2626"))
    fig.add_vline(x=(1-0.21)*1000, line_dash="dash", line_color="#16a34a",
                  annotation_text="Approval Threshold (score 790)")
    fig.update_layout(barmode="overlay",height=300,
                      xaxis_title="Risk Score (0=Riskiest, 1000=Safest)",
                      margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)


# ── PAGE 3: BATCH SCORING ─────────────────────────────────────────────────────
def page_batch_scoring():
    st.title("📋 Batch Credit Scoring")
    st.markdown("Upload a CSV → score all applications → download results.")

    template = pd.DataFrame([{
        "AMT_INCOME_TOTAL":135000,"AMT_CREDIT":406597,"AMT_ANNUITY":24700,
        "DAYS_BIRTH":-14235,"DAYS_EMPLOYED":-2160,
        "EXT_SOURCE_1":0.52,"EXT_SOURCE_2":0.64,"EXT_SOURCE_3":0.31,
        "CODE_GENDER":"M","FLAG_OWN_CAR":"Y","FLAG_OWN_REALTY":"N",
        "CNT_CHILDREN":1,"CNT_FAM_MEMBERS":3.0,
        "NAME_CONTRACT_TYPE":"Cash loans","NAME_INCOME_TYPE":"Working",
        "NAME_EDUCATION_TYPE":"Higher education",
    },{
        "AMT_INCOME_TOTAL":67500,"AMT_CREDIT":675000,"AMT_ANNUITY":33750,
        "DAYS_BIRTH":-9125,"DAYS_EMPLOYED":-365,
        "EXT_SOURCE_1":0.18,"EXT_SOURCE_2":0.22,"EXT_SOURCE_3":0.15,
        "CODE_GENDER":"M","FLAG_OWN_CAR":"N","FLAG_OWN_REALTY":"N",
        "CNT_CHILDREN":3,"CNT_FAM_MEMBERS":5.0,
        "NAME_CONTRACT_TYPE":"Cash loans","NAME_INCOME_TYPE":"Working",
        "NAME_EDUCATION_TYPE":"Secondary / secondary special",
    }])

    st.download_button("⬇️ Download Template CSV",
                       template.to_csv(index=False).encode(),
                       "template.csv","text/csv")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV (max 500 rows)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.info(f"Loaded {len(df):,} applications")
        if len(df) > 500:
            st.error("Max 500 rows per batch."); return
        required = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DAYS_BIRTH","DAYS_EMPLOYED"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing: {missing}"); return

        if st.button("⚡ Score All"):
            with st.spinner(f"Scoring {len(df)} applications..."):
                result = predict_batch(df.fillna(0).to_dict("records"))
            if not result:
                st.error("Batch failed."); return
            preds = result["predictions"]
            df["default_probability"] = [p["default_probability"] for p in preds]
            df["risk_score"]          = [p["risk_score"]          for p in preds]
            df["risk_label"]          = [p["risk_label"]          for p in preds]
            df["recommended_action"]  = [p["recommended_action"]  for p in preds]
            st.success(f"✅ Scored {len(df)} in {result['batch_latency_ms']:.0f}ms")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total",   len(df))
            c2.metric("Approve", (df["risk_label"]=="Low Risk").sum())
            c3.metric("Review",  df["risk_label"].isin(["Medium Risk","High Risk"]).sum())
            c4.metric("Decline", (df["risk_label"]=="Very High Risk").sum())
            dist = df["risk_label"].value_counts().reset_index()
            dist.columns = ["Risk Label","Count"]
            fig = px.pie(dist,values="Count",names="Risk Label",hole=0.4,
                         color_discrete_map={"Low Risk":"#16a34a","Medium Risk":"#d97706",
                                             "High Risk":"#dc2626","Very High Risk":"#7c3aed"})
            fig.update_layout(height=280,margin=dict(t=10,b=10))
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(df[["AMT_INCOME_TOTAL","AMT_CREDIT","default_probability",
                             "risk_score","risk_label","recommended_action"]]
                         .sort_values("default_probability",ascending=False),
                         use_container_width=True)
            st.download_button("⬇️ Download Results",
                               df.to_csv(index=False).encode(),
                               "scored.csv","text/csv")


# ── PAGE 4: RISK ANALYTICS ────────────────────────────────────────────────────
def page_risk_analytics():
    st.title("📈 Portfolio Risk Analytics")
    np.random.seed(42)
    scores = np.concatenate([np.random.beta(8,2,9193)*1000,
                             np.random.beta(2,6,807)*1000])
    labels = ["Low Risk" if s>=790 else "Medium Risk" if s>=650
              else "High Risk" if s>=500 else "Very High Risk" for s in scores]
    df_p = pd.DataFrame({"score":scores,"risk_label":labels})

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Portfolio Size",      "10,000")
    c2.metric("Avg Risk Score",      f"{scores.mean():.0f}")
    c3.metric("High+Very High Risk", f"{(df_p['risk_label'].isin(['High Risk','Very High Risk'])).mean()*100:.1f}%")
    c4.metric("Expected Default",    "~8.1%")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        dist = df_p["risk_label"].value_counts().reset_index()
        dist.columns = ["Risk Label","Count"]
        order = ["Low Risk","Medium Risk","High Risk","Very High Risk"]
        dist["Risk Label"] = pd.Categorical(dist["Risk Label"],categories=order,ordered=True)
        fig = px.bar(dist.sort_values("Risk Label"),x="Risk Label",y="Count",
                     color="Risk Label",
                     color_discrete_map={"Low Risk":"#16a34a","Medium Risk":"#d97706",
                                         "High Risk":"#dc2626","Very High Risk":"#7c3aed"},
                     title="Population by Risk Bucket",text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320,showlegend=False,margin=dict(t=40,b=10))
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores,nbinsx=60,
                                   marker_color="#2563EB",opacity=0.75))
        for thresh,color,lbl in [(790,"#16a34a","Approve"),(650,"#d97706","Review"),(500,"#dc2626","Decline")]:
            fig.add_vline(x=thresh,line_dash="dash",line_color=color,
                          annotation_text=lbl,annotation_position="top")
        fig.update_layout(title="Score Distribution",height=320,
                          xaxis_title="Risk Score (0=Riskiest, 1000=Safest)",
                          margin=dict(t=40,b=10))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.subheader("🎚️ Threshold Sensitivity")
    threshold = st.slider("Approval Threshold (Risk Score)",400,900,790,10)
    approved = (scores>=threshold).sum(); n=len(scores)
    c1,c2,c3 = st.columns(3)
    c1.metric("Approved",f"{approved:,}",f"{approved/n*100:.1f}%")
    c2.metric("Declined",f"{n-approved:,}",f"{(n-approved)/n*100:.1f}%")
    c3.metric("Est. Revenue",f"R${approved*25000/1e6:.1f}M","at avg R$25K loan")
    st.info(f"At threshold {threshold}: {approved/n*100:.1f}% approval rate. "
            "Lowering threshold = more approvals but higher expected defaults.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    page = sidebar()
    if   page == "🎯 Live Scoring":    page_live_scoring()
    elif page == "📊 Model Insights":  page_model_insights()
    elif page == "📋 Batch Scoring":   page_batch_scoring()
    elif page == "📈 Risk Analytics":  page_risk_analytics()

if __name__ == "__main__":
    main()
