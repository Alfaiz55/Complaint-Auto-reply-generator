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
    /* App background */
    .stApp {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, sans-serif;
    }

    /* Main container card */
    .main-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 26px 28px;
        border-radius: 16px;
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.08),
                    0 12px 40px rgba(2, 6, 23, 0.9);
    }

    /* Main title */
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #e0f2fe;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
    }

    /* Subtitle */
    .subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 22px;
    }

    /* Section headings */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #bae6fd;
        margin-top: 16px;
        margin-bottom: 6px;
    }

    /* Complaint reply output box */
    .reply-box {
        background: linear-gradient(
            145deg,
            rgba(14, 165, 233, 0.12),
            rgba(2, 132, 199, 0.08)
        );
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #e5e7eb;
        padding: 16px;
        border-radius: 12px;
        font-size: 15px;
        line-height: 1.55;
        box-shadow: inset 0 0 18px rgba(56, 189, 248, 0.15);
        margin-top: 10px;
    }

    /* Text area (complaint input) */
    textarea {
        background: rgba(2, 6, 23, 0.9) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 10px !important;
    }

    textarea::placeholder {
        color: #64748b !important;
    }

    /* Submit button */
    .stButton button {
        background: linear-gradient(135deg, #0ea5e9, #38bdf8);
        color: #020617;
        font-weight: 600;
        border-radius: 10px;
        padding: 8px 18px;
        border: none;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.35);
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(56, 189, 248, 0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)

        st.markdown("<div class='main-title'>Complaint Auto Reply Generator</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Submit your complaint and get quick assistance.</div>", unsafe_allow_html=True)

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
                        "reply": r["reply"],          # ✅ FIXED: reply is now saved
                        "extra_info": extra_info,
                    }

                    st.session_state.history.append(record)
                    pd.DataFrame(st.session_state.history).to_csv(HISTORY_CSV, index=False)

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



