# app.py – Complaint Auto Reply Generator (Final Stable Version)

import time
from pathlib import Path
from typing import Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Complaint Auto Reply Generator",
    layout="centered",
)

# =========================================================
# SESSION STATE
# =========================================================
if "role" not in st.session_state:
    st.session_state.role = None  # None | user | admin

if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# PATHS (GitHub / Streamlit safe)
# =========================================================
ART_DIR = Path("complaint_artifacts")
PIPELINE_PATH = ART_DIR / "pipeline_calibrated.joblib"
BANK_PATH = ART_DIR / "complaint_bank.pkl"
SBERT_META_PATH = ART_DIR / "sbert_meta.joblib"
BANK_EMB_PATH = ART_DIR / "bank_embeddings.npy"
HISTORY_CSV = ART_DIR / "complaint_history.csv"

# =========================================================
# STYLING (UNCHANGED USER UI)
# =========================================================
st.markdown(
    """
    <style>
    .stApp { background-color:#f3f6fb; }
    .main-card {
        background:#ffffff;
        padding:20px 22px;
        border-radius:12px;
        box-shadow:0 4px 14px rgba(15,23,42,0.08);
    }
    .main-title {
        font-size:28px;
        font-weight:700;
        color:#0f172a;
    }
    .subtitle {
        color:#6b7280;
        font-size:14px;
        margin-bottom:18px;
    }
    .section-title {
        font-size:18px;
        font-weight:600;
        color:#111827;
        margin-top:12px;
    }
    .reply-box {
        background:#e0f2fe;
        border:1px solid #93c5fd;
        color:#0f172a;
        padding:14px;
        border-radius:8px;
        margin-top:8px;
        line-height:1.5;
    }
    .meta-text {
        color:#6b7280;
        font-size:13px;
        margin-top:6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOADERS
# =========================================================
@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    if not path.exists():
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj

@st.cache_resource(show_spinner=False)
def load_bank(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    if SBERT_META_PATH.exists():
        meta = joblib.load(SBERT_META_PATH)
        name = meta.get("sbert_model_name", "all-MiniLM-L6-v2")
    else:
        name = "all-MiniLM-L6-v2"
    return SentenceTransformer(name)

pipeline = load_pipeline(PIPELINE_PATH)
bank = load_bank(BANK_PATH)
sbert = load_sbert()
bank_embs = np.load(BANK_EMB_PATH) if BANK_EMB_PATH.exists() else None

# =========================================================
# RULES
# =========================================================
LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "not delivered"],
    "billing": ["charged", "refund", "payment", "invoice"],
    "product": ["damaged", "defective", "broken", "wrong"],
    "account": ["login", "password", "otp", "account"],
    "technical": ["app", "error", "bug", "crash"],
}

def rule_override_label(text, model_label, conf):
    t = text.lower()
    hits = {k: 0 for k in LABEL_KEYWORDS}
    for k, kws in LABEL_KEYWORDS.items():
        for w in kws:
            if w in t:
                hits[k] += 1
    best = max(hits, key=hits.get)
    if hits[best] == 0:
        return model_label
    if conf and conf >= 0.80:
        return model_label
    return best

# =========================================================
# CORE INFERENCE (UNCHANGED)
# =========================================================
def get_reply(text):
    if sbert and bank and bank_embs is not None:
        q = sbert.encode([text], convert_to_numpy=True)
        sims = cosine_similarity(q, bank_embs)[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= 0.65:
            m = bank[idx]
            return m["label"], m["reply"], sims[idx]

    pred = pipeline.predict([text])[0]
    try:
        probs = pipeline.predict_proba([text])[0]
        conf = float(np.max(probs))
    except Exception:
        conf = None

    final_label = rule_override_label(text, pred, conf)

    replies = {
        "delivery": "We understand the delivery issue and will resolve it soon.",
        "billing": "We will review your billing concern and update you.",
        "product": "Your product issue is being reviewed.",
        "account": "We will assist you with your account issue.",
        "technical": "Our technical team will check this issue.",
    }

    return final_label, replies.get(final_label), conf

# =========================================================
# LOAD CSV HISTORY
# =========================================================
if not st.session_state.history and HISTORY_CSV.exists():
    st.session_state.history = pd.read_csv(HISTORY_CSV).to_dict("records")

# =========================================================
# SELECTION PAGE
# =========================================================
if st.session_state.role is None:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Complaint Auto Reply Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Fast, reliable complaint resolution</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("User", use_container_width=True):
            st.session_state.role = "user"
            st.rerun()
    with c2:
        if st.button("Admin", use_container_width=True):
            st.session_state.role = "admin"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================================================
# ADMIN LOGIN
# =========================================================
if st.session_state.role == "admin" and not st.session_state.admin_logged:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Admin Login</div>", unsafe_allow_html=True)

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "0000":
            st.session_state.admin_logged = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================================================
# USER PANEL (UNCHANGED UI)
# =========================================================
if st.session_state.role == "user":
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Complaint Auto-Responder</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Enter your complaint below</div>", unsafe_allow_html=True)

        complaint = st.text_area("Complaint", height=150)
        if st.button("Submit", type="primary"):
            if complaint.strip():
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

                st.markdown("<div class='section-title'>Response</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='reply-box'>{reply}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ADMIN PANEL (UNCHANGED UI)
# =========================================================
if st.session_state.role == "admin" and st.session_state.admin_logged:
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.history)
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
                "Download Complaint CSV",
                data=df.to_csv(index=False),
                file_name="complaint_history.csv",
                mime="text/csv",
            )

        st.markdown("</div>", unsafe_allow_html=True)
