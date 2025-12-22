# app.py – Clean complaint UI with User/Admin panels, dashboard, history, rules + CSV persistence
# FINAL STABLE VERSION (Streamlit Cloud compatible)

import time
from pathlib import Path
from typing import Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Complaint Auto-Responder",
    layout="centered",
)

# ---------------- Paths (GitHub safe) ----------------
BASE_DIR = Path(__file__).parent
ART_DIR = BASE_DIR / "complaint_artifacts"

PIPELINE_PATH = ART_DIR / "pipeline_calibrated.joblib"
BANK_PATH = ART_DIR / "complaint_bank.pkl"
SBERT_META_PATH = ART_DIR / "sbert_meta.joblib"
BANK_EMB_PATH = ART_DIR / "bank_embeddings.npy"
HISTORY_CSV = ART_DIR / "complaint_history.csv"

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

# ---------------- Loaders ----------------
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

# ---------------- Rule Keywords (UNCHANGED) ----------------
LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "not delivered", "late delivery"],
    "billing": ["charged", "refund", "payment", "invoice"],
    "product": ["damaged", "broken", "defective", "wrong product"],
    "account": ["login", "password", "otp", "account"],
    "technical": ["app", "bug", "error", "crash"],
}

def rule_override_label(text: str, model_label: str, conf: Optional[float]) -> str:
    t = text.lower()
    hits = {k: sum(kw in t for kw in v) for k, v in LABEL_KEYWORDS.items()}
    best = max(hits, key=hits.get)
    if hits[best] == 0:
        return model_label
    if conf is not None and conf >= 0.80:
        return model_label
    return best

# ---------------- Core Logic ----------------
def get_reply(text: str, sim_threshold: float = 0.65):

    if pipeline is None:
        return {
            "method": "error",
            "label": None,
            "reply": "Model not loaded correctly.",
            "confidence": None,
        }

    if sbert is not None and bank is not None and bank_embs is not None:
        q = sbert.encode([text], convert_to_numpy=True)
        sims = cosine_similarity(q, bank_embs)[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= sim_threshold:
            m = bank[idx]
            return {
                "method": "retrieval",
                "label": m["label"],
                "reply": m["reply"],
                "confidence": float(sims[idx]),
            }

    pred = pipeline.predict([text])[0]
    try:
        probs = pipeline.predict_proba([text])[0]
        conf = float(np.max(probs))
    except Exception:
        conf = None

    final_label = rule_override_label(text, pred, conf)

    default_replies = {
        "billing": "Thanks for telling us — we understand. We'll check your billing issue and update you soon.",
        "delivery": "Thanks for telling us — we understand. We'll investigate the delivery issue and update you soon.",
        "product": "Thanks for telling us — we understand. We'll review the product issue and update you soon.",
        "account": "Thanks for telling us — we understand. We'll resolve your account issue soon.",
        "technical": "Thanks for telling us — we understand. Our technical team will check this issue.",
    }

    return {
        "method": "classifier",
        "label": final_label,
        "reply": default_replies.get(final_label),
        "confidence": conf,
    }

# ---------------- Persistent History ----------------
if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        st.session_state["history"] = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        st.session_state["history"] = []

# ---------------- Sidebar (UNCHANGED) ----------------
mode = st.sidebar.radio("View as", ["User panel", "Admin panel"], index=0)

# ---------------- USER PANEL (UNCHANGED UI) ----------------
if mode == "User panel":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Complaint Auto-Responder</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Enter your complaint related to our services.</div>",
        unsafe_allow_html=True,
    )

    complaint = st.text_area(
        "Enter the complaint",
        height=150,
        placeholder="Example: The delivery boy asked me to come to the office...",
    )

    if st.button("Submit", type="primary"):
        if complaint.strip():
            result = get_reply(complaint.strip())
            if result["method"] == "error":
                st.error(result["reply"])
            else:
                record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "complaint": complaint,
                    "label": result["label"],
                    "method": result["method"],
                    "confidence": result["confidence"],
                }
                st.session_state["history"].append(record)
                pd.DataFrame([record]).to_csv(
                    HISTORY_CSV,
                    mode="a",
                    index=False,
                    header=not HISTORY_CSV.exists(),
                )

                st.markdown(
                    f"<div class='reply-box'><b>We received your complaint related to {result['label']}</b><br>{result['reply']}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ADMIN PANEL (UNCHANGED UI) ----------------
else:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)

    if st.session_state["history"]:
        df = pd.DataFrame(st.session_state["history"])
        st.write(f"Total complaints: {len(df)}")

        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]

        chart = alt.Chart(counts).mark_arc().encode(
            theta="count", color="label", tooltip=["label", "count"]
        )
        st.altair_chart(chart, width="stretch")

        st.dataframe(df, width="stretch")
    else:
        st.info("No complaints yet.")
    st.markdown("</div>", unsafe_allow_html=True)
