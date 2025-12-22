# app.py – Complaint Auto Reply Generator (STABLE VERSION)

import time
from pathlib import Path
from typing import Optional, Dict, List

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

# ---------------- Rule-based Keywords ----------------
LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "not delivered", "late delivery"],
    "billing": ["refund", "billing", "charged", "payment", "invoice"],
    "product": ["damaged", "broken", "defective", "wrong product"],
    "account": ["login", "password", "otp", "account"],
    "technical": ["app", "error", "crash", "bug", "not working"],
}


def rule_override_label(text: str, model_label: str, conf: Optional[float]):
    text = text.lower()
    scores = {k: 0 for k in LABEL_KEYWORDS}

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


# ---------------- Core Inference (FIXED) ----------------
def get_reply(text: str):
    if pipeline is None:
        return {
            "method": "error",
            "label": "unknown",
            "reply": "Model not loaded correctly.",
            "confidence": None,
        }

    pred = pipeline.predict([text])[0]

    try:
        probs = pipeline.predict_proba([text])[0]
        conf = float(np.max(probs))
    except Exception:
        conf = None

    final_label = rule_override_label(text, pred, conf)

    replies = {
        "billing": "Thanks for telling us — we understand. We'll check your billing issue and update you soon.",
        "delivery": "Thanks for telling us — we understand. We'll investigate the delivery issue and update you soon.",
        "product": "Thanks for telling us — we understand. We'll review the product issue and resolve it shortly.",
        "account": "Thanks for telling us — we understand. We'll assist you with your account issue shortly.",
        "technical": "Thanks for telling us — we understand. Our technical team will look into this issue.",
    }

    reply = replies.get(
        final_label,
        "Thanks for telling us — we understand. We'll look into this and update you soon.",
    )

    return {
        "method": "classifier",
        "label": final_label,
        "reply": reply,
        "confidence": conf,
    }


# ---------------- Persistent History ----------------
if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        st.session_state.history = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        st.session_state.history = []


# ---------------- Sidebar ----------------
mode = st.sidebar.radio("View as", ["User panel", "Admin panel"], index=0)

st.sidebar.markdown("---")
st.sidebar.write("Model loaded:", pipeline is not None)

# ================= USER PANEL =================
if mode == "User panel":
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Complaint Auto Reply Generator</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Enter your complaint related to our services. "
            "We will provide a quick and helpful response.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-title'>Enter the complaint</div>", unsafe_allow_html=True)
        complaint = st.text_area(
            "Complaint",
            height=150,
            placeholder="Example: The delivery boy asked me to collect the product from office.",
        )

        if st.button("Submit", type="primary"):
            if not complaint.strip():
                st.warning("Please enter a complaint.")
            else:
                with st.spinner("Analyzing complaint..."):
                    start = time.time()
                    result = get_reply(complaint.strip())
                    elapsed = time.time() - start

                if result["method"] == "error":
                    st.error(result["reply"])
                else:
                    record = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "complaint": complaint.strip(),
                        "label": result["label"],
                        "method": result["method"],
                        "confidence": result["confidence"],
                        "reply": result["reply"],
                    }

                    st.session_state.history.append(record)
                    pd.DataFrame(st.session_state.history).to_csv(
                        HISTORY_CSV, index=False
                    )

                    st.markdown("<div class='section-title'>Suggested response</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='reply-box'>We received your complaint related to "
                        f"<b>{result['label']}</b> :-<br>{result['reply']}</div>",
                        unsafe_allow_html=True,
                    )

                    meta = []
                    if result["confidence"] is not None:
                        meta.append(f"Confidence: {result['confidence']:.2f}")
                    meta.append(f"Generated in {elapsed:.2f}s")

                    st.markdown(
                        "<div class='meta-text'>" + " • ".join(meta) + "</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("</div>", unsafe_allow_html=True)

# ================= ADMIN PANEL =================
else:
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Statistics and history of processed complaints.</div>",
            unsafe_allow_html=True,
        )

        history = st.session_state.history

        if not history:
            st.info("No complaints processed yet.")
        else:
            df = pd.DataFrame(history)

            st.markdown("<div class='section-title'>Complaint Dashboard</div>", unsafe_allow_html=True)
            st.write(f"Total complaints: **{len(df)}**")

            counts = df["label"].value_counts().reset_index()
            counts.columns = ["label", "count"]

            chart = alt.Chart(counts).mark_arc().encode(
                theta="count",
                color="label",
                tooltip=["label", "count"],
            )
            st.altair_chart(chart, width="stretch")

            st.markdown("<div class='section-title'>Complaint History</div>", unsafe_allow_html=True)
            st.dataframe(df, width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)
