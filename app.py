# app.py – Complaint Auto Reply Generator (User/Admin separated)

import time
from pathlib import Path
from typing import Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Complaint Auto Reply Generator",
    layout="centered",
)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).parent
PIPELINE_PATH = BASE_DIR / "pipeline_calibrated.joblib"
BANK_PATH = BASE_DIR / "complaint_bank.pkl"
SBERT_META_PATH = BASE_DIR / "sbert_meta.joblib"
BANK_EMB_PATH = BASE_DIR / "bank_embeddings.npy"
HISTORY_CSV = BASE_DIR / "complaint_history.csv"

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
    .stApp { background-color:#f3f6fb; }

    .main-card {
        background:#ffffff;
        padding:22px;
        border-radius:12px;
        box-shadow:0 4px 14px rgba(15,23,42,0.08);
        margin-top:20px;
    }

    .moto {
        background:#2563eb;
        color:white;
        padding:14px;
        border-radius:10px;
        text-align:center;
        font-size:20px;
        font-weight:600;
        margin-bottom:20px;
    }

    .subtitle {
        color:#6b7280;
        font-size:14px;
        margin-bottom:18px;
    }

    .section-title {
        font-size:18px;
        font-weight:600;
        margin-top:12px;
        margin-bottom:6px;
        color:#111827;
    }

    .reply-box {
        background:#e0f2fe;
        border:1px solid #93c5fd;
        color:#0f172a;
        padding:14px;
        border-radius:8px;
        margin-top:10px;
        font-size:15px;
    }

    .meta-text {
        font-size:13px;
        color:#6b7280;
        margin-top:6px;
    }

    .center-btn {
        display:flex;
        justify-content:center;
        gap:20px;
        margin-top:20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Loaders ----------------
@st.cache_resource
def load_pipeline(path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_bank(path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        meta = joblib.load(SBERT_META_PATH)
        return SentenceTransformer(meta.get("sbert_model_name"))
    except:
        return None

pipeline = load_pipeline(PIPELINE_PATH)
bank = load_bank(BANK_PATH)
sbert = load_sbert()
bank_embs = np.load(BANK_EMB_PATH) if BANK_EMB_PATH.exists() else None

# ---------------- Session ----------------
if "page" not in st.session_state:
    st.session_state.page = "select"

if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        st.session_state.history = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        st.session_state.history = []

# ---------------- Label Rules ----------------
LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "not delivered", "delivery boy"],
    "billing": ["charged", "refund", "payment", "invoice"],
    "product": ["damaged", "broken", "wrong product", "quality"],
    "account": ["login", "password", "account", "otp"],
    "technical": ["crash", "error", "not working", "bug"],
}

def rule_override(text, pred):
    t = text.lower()
    for label, kws in LABEL_KEYWORDS.items():
        for k in kws:
            if k in t:
                return label
    return pred

# ---------------- Inference ----------------
def get_reply(text):
    if sbert and bank and bank_embs is not None:
        q = sbert.encode([text], convert_to_numpy=True)
        sims = cosine_similarity(q, bank_embs)[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= 0.65:
            b = bank[idx]
            return b["label"], b["reply"], "retrieval", sims[idx]

    pred = pipeline.predict([text])[0]
    final = rule_override(text, pred)

    replies = {
        "delivery": "We understand your delivery concern. Our team will investigate and update you shortly.",
        "billing": "We understand the billing issue. Our billing team will review and get back to you.",
        "product": "We understand the product issue. We will arrange a resolution as soon as possible.",
        "account": "We understand your account problem. Our support team will assist you shortly.",
        "technical": "We understand the technical issue. Our engineers are checking this.",
    }

    return final, replies.get(final), "classifier", None

# ================= PAGE 1: SELECTION =================
if st.session_state.page == "select":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='moto'>Complain here — we provide quick and reliable services</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Choose how you want to continue</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if col1.button("User", use_container_width=True):
        st.session_state.page = "user"
        st.rerun()

    if col2.button("Admin", use_container_width=True):
        st.session_state.page = "admin_login"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ================= PAGE 2: ADMIN LOGIN =================
elif st.session_state.page == "admin_login":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='moto'>Admin Login</div>", unsafe_allow_html=True)

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if u == "admin" and p == "0000":
            st.session_state.page = "admin"
            st.rerun()
        else:
            st.error("Invalid admin credentials")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= PAGE 3: USER =================
elif st.session_state.page == "user":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='moto'>Submit Your Complaint</div>", unsafe_allow_html=True)

    complaint = st.text_area("Enter your complaint", height=150)

    if st.button("Submit", use_container_width=True):
        if complaint.strip():
            label, reply, method, conf = get_reply(complaint)

            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "complaint": complaint,
                "label": label,
                "method": method,
                "confidence": conf,
                "reply": reply,
            }

            st.session_state.history.append(record)
            pd.DataFrame([record]).to_csv(
                HISTORY_CSV,
                mode="a",
                index=False,
                header=not HISTORY_CSV.exists(),
            )

            st.markdown(
                f"<div class='reply-box'>We received your complaint related to <b>{label}</b> :-<br>{reply}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='meta-text'>Thank you for contacting us. We appreciate your patience.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Please enter a complaint")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= PAGE 4: ADMIN =================
elif st.session_state.page == "admin":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='moto'>Admin Dashboard</div>", unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state.history)

    if df.empty:
        st.info("No complaints available")
    else:
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)

        chart = (
            alt.Chart(counts)
            .mark_arc()
            .encode(
                theta="count",
                color=alt.Color("label", legend=alt.Legend(orient="left")),
                tooltip=["label", "count", "percentage"],
            )
        )

        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Complaint CSV",
            df.to_csv(index=False),
            file_name="complaint_history.csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)
