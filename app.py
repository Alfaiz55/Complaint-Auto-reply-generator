# app.py – Complaint Auto Reply Generator (FINAL STABLE VERSION)

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

# ---------------- Paths (GitHub-safe) ----------------
BASE_DIR = Path(__file__).parent
PIPELINE_PATH = BASE_DIR / "pipeline_calibrated.joblib"
HISTORY_CSV = BASE_DIR / "complaint_history.csv"

# ---------------- UI Styling (UNCHANGED) ----------------
st.markdown(
    """
    <style>
    .stApp { background-color: #f3f6fb; }
    .main-card {
        background: #ffffff;
        padding: 20px 22px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
    }
    .main-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
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
        margin-top: 12px;
        margin-bottom: 6px;
        color: #111827;
    }
    .reply-box {
        background: #e0f2fe;
        border: 1px solid #93c5fd;
        color: #0f172a;
        padding: 14px;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.5;
        margin-top: 8px;
    }
    .meta-text {
        color: #6b7280;
        font-size: 13px;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Load ML Pipeline ----------------
@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    if not path.exists():
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj

pipeline = load_pipeline(PIPELINE_PATH)

# ---------------- Labels ----------------
ALL_LABELS = ["billing", "delivery", "product", "account", "technical"]

LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "not delivered", "late"],
    "billing": ["refund", "billing", "charged", "payment", "invoice"],
    "product": ["damaged", "broken", "defective", "wrong"],
    "account": ["login", "password", "otp", "account"],
    "technical": ["app", "error", "crash", "bug"],
}

def rule_override_label(text: str, model_label: str, conf: Optional[float]):
    text = text.lower()
    scores = {k: 0 for k in ALL_LABELS}
    for label, words in LABEL_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[label] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return model_label
    if conf is not None and conf >= 0.80:
        return model_label
    return best

# ---------------- Inference ----------------
def get_reply(text: str):
    if pipeline is None:
        return None

    pred = pipeline.predict([text])[0]
    try:
        conf = float(np.max(pipeline.predict_proba([text])[0]))
    except Exception:
        conf = None

    label = rule_override_label(text, pred, conf)

    replies = {
        "billing": "Thanks for telling us — we understand. We'll check your billing issue and update you soon.",
        "delivery": "Thanks for telling us — we understand. We'll investigate the delivery issue and update you soon.",
        "product": "Thanks for telling us — we understand. We'll review the product issue and resolve it shortly.",
        "account": "Thanks for telling us — we understand. We'll assist you with your account issue shortly.",
        "technical": "Thanks for telling us — we understand. Our technical team will look into this issue.",
    }

    return label, replies[label], conf

# ---------------- History ----------------
if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        st.session_state.history = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        st.session_state.history = []

# ---------------- Sidebar ----------------
mode = st.sidebar.radio("View as", ["User panel", "Admin panel"], index=0)

# ================= USER PANEL =================
if mode == "User panel":
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Complaint Auto Reply Generator</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Enter your complaint related to our services.</div>", unsafe_allow_html=True)

        complaint = st.text_area("Complaint", height=150)

        if st.button("Submit", type="primary"):
            if not complaint.strip():
                st.warning("Please enter a complaint.")
            else:
                result = get_reply(complaint.strip())
                if result is None:
                    st.error("Model not loaded correctly.")
                else:
                    label, reply, conf = result
                    st.session_state.predicted_label = label
                    st.session_state.auto_reply = reply
                    st.session_state.confidence = conf
                    st.session_state.show_extra = True

        if st.session_state.get("show_extra"):
            st.markdown(
                f"<div class='reply-box'><b>Auto response ({st.session_state.predicted_label}):</b><br>"
                f"{st.session_state.auto_reply}</div>",
                unsafe_allow_html=True,
            )

            extra_labels = {
                "billing": "Billing ID",
                "delivery": "Order / Tracking ID",
                "product": "Order ID",
                "account": "Username or Email",
                "technical": "Device & App version",
            }

            extra_info = st.text_input(
                f"Please enter {extra_labels[st.session_state.predicted_label]}"
            )

            if st.button("Submit Complaint"):
                if not extra_info.strip():
                    st.warning("This field is required.")
                else:
                    record = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "complaint": complaint,
                        "label": st.session_state.predicted_label,
                        "extra_info": extra_info,
                        "reply": st.session_state.auto_reply,
                        "confidence": st.session_state.confidence,
                    }
                    st.session_state.history.append(record)
                    pd.DataFrame(st.session_state.history).to_csv(HISTORY_CSV, index=False)
                    st.success("Your complaint has been registered successfully. Our team will resolve it shortly.")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= ADMIN PANEL =================
else:
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)

        if not st.session_state.history:
            st.info("No complaints processed yet.")
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
            st.dataframe(df, width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)
