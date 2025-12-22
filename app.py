# app.py – Complaint Auto Reply Generator (User/Admin with Login)

import time
from pathlib import Path
from typing import Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Complaint Auto Reply Generator",
    layout="centered",
)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).parent

PIPELINE_PATH   = BASE_DIR / "pipeline_calibrated.joblib"
BANK_PATH       = BASE_DIR / "complaint_bank.pkl"
SBERT_META_PATH = BASE_DIR / "sbert_meta.joblib"
BANK_EMB_PATH   = BASE_DIR / "bank_embeddings.npy"
HISTORY_CSV     = BASE_DIR / "complaint_history.csv"

# ---------------- UI Styling ----------------
st.markdown("""
<style>
.stApp { background-color: #f3f6fb; }
.card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
.title { font-size: 26px; font-weight: 700; }
.subtitle { color: #6b7280; margin-bottom: 16px; }
.reply-box {
    background: #e0f2fe;
    border: 1px solid #93c5fd;
    padding: 14px;
    border-radius: 8px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Loaders ----------------
@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    if not path.exists():
        return None
    obj = joblib.load(path)
    return obj["pipeline"] if isinstance(obj, dict) else obj

@st.cache_resource(show_spinner=False)
def load_bank(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource(show_spinner=False)
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        name = "all-MiniLM-L6-v2"
        if SBERT_META_PATH.exists():
            meta = joblib.load(SBERT_META_PATH)
            name = meta.get("sbert_model_name", name)
        return SentenceTransformer(name)
    except Exception:
        return None

pipeline = load_pipeline(PIPELINE_PATH)
bank = load_bank(BANK_PATH)
sbert = load_sbert()
bank_embs = np.load(BANK_EMB_PATH) if BANK_EMB_PATH.exists() else None

# ---------------- Label Rules ----------------
LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "not delivered"],
    "billing": ["refund", "charged", "payment", "invoice"],
    "product": ["damaged", "broken", "wrong product", "defective"],
    "account": ["login", "password", "otp", "account"],
    "technical": ["error", "crash", "bug", "not working"],
}

def rule_override(text, label, conf):
    t = text.lower()
    scores = {k: sum(kw in t for kw in v) for k, v in LABEL_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0 and (conf is None or conf < 0.8):
        return best
    return label

# ---------------- Inference ----------------
def get_reply(text):
    if sbert and bank and bank_embs is not None:
        q = sbert.encode([text], convert_to_numpy=True)
        sims = cosine_similarity(q, bank_embs)[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= 0.65:
            m = bank[idx]
            return m["label"], m["reply"], "retrieval", float(sims[idx])

    pred = pipeline.predict([text])[0]
    try:
        conf = float(np.max(pipeline.predict_proba([text])[0]))
    except:
        conf = None

    label = rule_override(text, pred, conf)

    replies = {
        "delivery": "We understand your delivery concern. Our team will investigate and update you shortly.",
        "billing": "We understand the billing issue and will resolve it as soon as possible.",
        "product": "We acknowledge the product issue and will take necessary action.",
        "account": "We are reviewing your account-related concern and will update you.",
        "technical": "Our technical team is looking into the issue.",
    }

    return label, replies.get(label, "We have received your complaint."), "classifier", conf

# ---------------- Session Init ----------------
if "role" not in st.session_state:
    st.session_state.role = None

# ---------------- LOGIN PAGE ----------------
if st.session_state.role is None:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>Complaint Auto Reply Generator</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Select your access type</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        if col1.button("User"):
            st.session_state.role = "user"
            st.experimental_rerun()

        if col2.button("Admin"):
            st.session_state.role = "admin_login"
            st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ADMIN LOGIN ----------------
elif st.session_state.role == "admin_login":
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>Admin Login</div>", unsafe_allow_html=True)

        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u == "admin" and p == "0000":
                st.session_state.role = "admin"
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- USER PANEL ----------------
elif st.session_state.role == "user":
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>Submit Your Complaint</div>", unsafe_allow_html=True)

        complaint = st.text_area("Complaint", height=150)

        if st.button("Submit Complaint"):
            if complaint.strip():
                label, reply, method, conf = get_reply(complaint)

                record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "complaint": complaint,
                    "label": label,
                    "method": method,
                    "confidence": conf,
                }

                df = pd.DataFrame([record])
                df.to_csv(HISTORY_CSV, mode="a", index=False, header=not HISTORY_CSV.exists())

                st.markdown(f"<div class='reply-box'><b>Related to:</b> {label}<br>{reply}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ADMIN PANEL ----------------
elif st.session_state.role == "admin":
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>Admin Dashboard</div>", unsafe_allow_html=True)

        if not HISTORY_CSV.exists():
            st.info("No complaints yet.")
        else:
            df = pd.read_csv(HISTORY_CSV)

            st.write(f"Total complaints: **{len(df)}**")

            counts = df["label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)

            chart = alt.Chart(counts).mark_arc().encode(
                theta="count",
                color=alt.Color("label", legend=alt.Legend(orient="left")),
                tooltip=["label", "count", "percentage"]
            )

            text = alt.Chart(counts).mark_text(radius=120).encode(
                theta="count",
                text=alt.Text("percentage:Q", format=".0f")
            )

            st.altair_chart(chart + text, use_container_width=True)

            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name="complaint_history.csv",
                mime="text/csv"
            )

        st.markdown("</div>", unsafe_allow_html=True)
