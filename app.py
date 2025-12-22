# app.py – Complaint Auto Reply Generator (FINAL STABLE VERSION)

import time
from pathlib import Path
from typing import Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Complaint Auto Reply Generator",
    layout="centered",
)

# --------------------------------------------------
# Paths (Cloud-safe)
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
PIPELINE_PATH = BASE_DIR / "pipeline_calibrated.joblib"
BANK_PATH = BASE_DIR / "complaint_bank.pkl"
SBERT_META_PATH = BASE_DIR / "sbert_meta.joblib"
BANK_EMB_PATH = BASE_DIR / "bank_embeddings.npy"
HISTORY_CSV = BASE_DIR / "complaint_history.csv"

# --------------------------------------------------
# UI Styling (UNCHANGED)
# --------------------------------------------------
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
    .main-title { font-size: 28px; font-weight: 700; color: #0f172a; }
    .subtitle { color: #6b7280; font-size: 14px; margin-bottom: 18px; }
    .section-title { font-size: 18px; font-weight: 600; color: #111827; }
    .reply-box {
        background: #e0f2fe;
        border: 1px solid #93c5fd;
        color: #0f172a;
        padding: 14px;
        border-radius: 8px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Session state init
# --------------------------------------------------
if "role" not in st.session_state:
    st.session_state.role = None
if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False
if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        try:
            st.session_state.history = pd.read_csv(HISTORY_CSV).to_dict("records")
        except:
            st.session_state.history = []
    else:
        st.session_state.history = []

# --------------------------------------------------
# Loaders
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return joblib.load(PIPELINE_PATH) if PIPELINE_PATH.exists() else None

@st.cache_resource(show_spinner=False)
def load_bank():
    return joblib.load(BANK_PATH) if BANK_PATH.exists() else None

@st.cache_resource(show_spinner=False)
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None

pipeline = load_pipeline()
bank = load_bank()
sbert = load_sbert()
bank_embs = np.load(BANK_EMB_PATH) if BANK_EMB_PATH.exists() else None

# --------------------------------------------------
# Prediction logic (UNCHANGED)
# --------------------------------------------------
def get_reply(text: str):
    if sbert and bank and bank_embs is not None:
        q = sbert.encode([text], convert_to_numpy=True)
        sims = cosine_similarity(q, bank_embs)[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= 0.65:
            return bank[idx]["label"], bank[idx]["reply"], sims[idx]

    pred = pipeline.predict([text])[0]
    conf = float(np.max(pipeline.predict_proba([text])[0]))
    default_reply = f"Thanks for telling us — we understand. We’ll handle your {pred} issue shortly."
    return pred, default_reply, conf

# --------------------------------------------------
# PAGE 1: ROLE SELECTION
# --------------------------------------------------
if st.session_state.role is None:
    st.markdown("<h2 style='text-align:center'>Complaint Auto Reply Generator</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555'>Quick complaint handling with intelligent automation</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("User", use_container_width=True):
            st.session_state.role = "user"
            st.rerun()
    with col2:
        if st.button("Admin", use_container_width=True):
            st.session_state.role = "admin"
            st.rerun()

# --------------------------------------------------
# ADMIN LOGIN
# --------------------------------------------------
elif st.session_state.role == "admin" and not st.session_state.admin_logged:
    st.markdown("### Admin Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "0000":
            st.session_state.admin_logged = True
            st.rerun()
        else:
            st.error("Invalid credentials")

# --------------------------------------------------
# USER PANEL (UNCHANGED UI)
# --------------------------------------------------
elif st.session_state.role == "user":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Complaint Auto-Responder</div>", unsafe_allow_html=True)

    complaint = st.text_area(
        "Enter the complaint",
        height=150,
        placeholder="Example: The delivery boy did not come to my home for delivery.",
    )

    if st.button("Submit", type="primary"):
        label, reply, conf = get_reply(complaint)

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complaint": complaint,
            "label": label,
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

        st.markdown(f"<div class='reply-box'><b>{label}</b>: {reply}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ADMIN PANEL (UNCHANGED UI + CSV DOWNLOAD)
# --------------------------------------------------
elif st.session_state.role == "admin" and st.session_state.admin_logged:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state.history)
    st.write(f"Total complaints: {len(df)}")

    if not df.empty:
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]

        chart = alt.Chart(counts).mark_arc().encode(
            theta="count",
            color="label",
            tooltip=["label", "count"]
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Complaint History CSV",
            data=df.to_csv(index=False),
            file_name="complaint_history.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)
