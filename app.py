# app.py – Complaint Auto Reply Generator (FINAL STABLE & CORRECTED VERSION)

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Complaint Auto Reply Generator",
    layout="centered",
)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).parent
PIPELINE_PATH = BASE_DIR / "pipeline_calibrated.joblib"
HISTORY_CSV = BASE_DIR / "complaint_history.csv"

# ---------------- UI Styling ----------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f3f6fb;
        }

        .main-card {
            background: #ffffff;
            padding: 20px 22px;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        }

        .main-title {
            font-size: 28px;
            font-weight: 700;
            color: #0f172a;
        }

        .subtitle {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 18px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #111827;
            margin-top: 12px;
        }
           .badge-row {
        display: flex;
        gap: 10px;
        margin-bottom: 18px;
        flex-wrap: wrap;
    }

    .badge {
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #0f172a;
        box-shadow: inset 0 0 10px rgba(56, 189, 248, 0.15);
    }

        .reply-box {
            background: #e0f2fe;
            border: 1px solid #93c5fd;
            color: #0f172a;
            padding: 14px;
            border-radius: 8px;
            font-size: 15px;
            margin-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


  
 

   
st.markdown("""
<div class="badge-row">
    <div class="badge">Product</div>
    <div class="badge">Account</div>
    <div class="badge">Billing</div>
    <div class="badge">Technical</div>
    <div class="badge">Delivery</div>
</div>
""", unsafe_allow_html=True)



# ---------------- Load Model ----------------
@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    if not path.exists():
        return None
    obj = joblib.load(path)
    return obj["pipeline"] if isinstance(obj, dict) and "pipeline" in obj else obj

pipeline = load_pipeline(PIPELINE_PATH)

# ---------------- Labels ----------------
ALL_LABELS = ["billing", "delivery", "product", "account", "technical"]

LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "late"],
    "billing": ["refund", "billing", "charged", "payment"],
    "product": ["damaged", "broken", "defective", "wrong"],
    "account": ["login", "password", "otp"],
    "technical": ["app", "error", "crash", "bug"],
}

def rule_override_label(text: str, model_label: str, conf: Optional[float]):
    scores = {k: 0 for k in ALL_LABELS}
    text = text.lower()
    for label, words in LABEL_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[label] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0 or (conf is not None and conf >= 0.80):
        return model_label
    return best

# ---------------- Inference ----------------
def get_reply(text: str):
    if pipeline is None:
        return {"error": True, "reply": "Model not loaded correctly."}

    pred = pipeline.predict([text])[0]
    try:
        conf = float(np.max(pipeline.predict_proba([text])[0]))
    except Exception:
        conf = None

    label = rule_override_label(text, pred, conf)

    replies = {
        "billing": "We will review your billing issue and update you shortly.",
        "delivery": "We are checking the delivery issue and will resolve it soon.",
        "product": "We will review the product issue and assist you.",
        "account": "We will help you resolve your account issue.",
        "technical": "Our technical team will look into this issue.",
    }

    return {
        "error": False,
        "label": label,
        "reply": replies.get(label, "We will review your issue."),
        "confidence": conf,
    }

# ---------------- History ----------------
if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        st.session_state.history = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        st.session_state.history = []

# ---------------- Sidebar ----------------
mode = st.sidebar.radio("View as", ["User panel", "Admin panel"])

# ================= USER PANEL =================
if mode == "User panel":

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='main-title'>Complaint Auto Reply Generator</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Submit your complaint and get quick assistance.</div>",
        unsafe_allow_html=True,
    )

    complaint = st.text_area(
        "Enter your complaint",
        height=150,
        placeholder="Example: The delivery boy asked me to collect the product from office.",
    )

    if st.button("Submit", type="primary"):
        if not complaint.strip():
            st.warning("Please enter a complaint.")
        else:
            st.session_state.result = get_reply(complaint.strip())
            st.session_state.complaint_text = complaint.strip()

    if "result" in st.session_state and not st.session_state.result["error"]:
        r = st.session_state.result

        st.markdown("<div class='section-title'>Suggested response</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='reply-box'>Complaint category: <b>{r['label']}</b><br>{r['reply']}</div>",
            unsafe_allow_html=True,
        )

        placeholders = {
            "billing": "e.g. BILL-2024-1098",
            "product": "e.g. ORD-458921",
            "delivery": "e.g. TRK-992134",
            "account": "e.g. registered email",
            "technical": "e.g. Android 13, App v2.4.1",
        }

        extra_info = st.text_input(
            "Additional information (required)",
            placeholder=placeholders[r["label"]],
        )

        if st.button("Submit Complaint"):
            if not extra_info.strip():
                st.warning("Please provide the required information.")
            else:
                record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "complaint": st.session_state.complaint_text,
                    "label": r["label"],
                    "confidence": r["confidence"],
                    "reply": r["reply"],
                    "extra_info": extra_info,
                }

                # Always treat CSV as source of truth
if HISTORY_CSV.exists():
    df_existing = pd.read_csv(HISTORY_CSV)
else:
    df_existing = pd.DataFrame()

# Append new record
df_updated = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)

# Save back to CSV
df_updated.to_csv(HISTORY_CSV, index=False)

# Refresh session_state from CSV
st.session_state.history = df_updated.to_dict("records")

st.success("Thank you. Your complaint has been registered successfully.")
del st.session_state.result

    st.markdown("</div>", unsafe_allow_html=True)


# ================= ADMIN PANEL =================
else:
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)

        if not st.session_state.history:
            st.info("No complaints yet.")
        else:
            df = pd.DataFrame(st.session_state.history)

            counts = df["label"].value_counts().reindex(ALL_LABELS, fill_value=0).reset_index()
            counts.columns = ["label", "count"]
            counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(1)

            chart = alt.Chart(counts).mark_arc().encode(
                theta="count",
                color="label",
                tooltip=["label", "count", "percentage"],
            )

            st.altair_chart(chart, width="stretch")
            df_display = df.copy().reset_index(drop=True)
            df_display.insert(0, "S.No", range(1, len(df_display) + 1))

            st.dataframe(df_display, width="stretch", hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)












