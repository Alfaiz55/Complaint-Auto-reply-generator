# app.py – Clean complaint UI with User/Admin panels, dashboard, history, rules + CSV persistence

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

# ---------------- Paths (GitHub ROOT) ----------------
PIPELINE_PATH = Path("pipeline_calibrated.joblib")
BANK_PATH = Path("complaint_bank.pkl")
SBERT_META_PATH = Path("sbert_meta.joblib")
BANK_EMB_PATH = Path("embeddings_full.npy")
HISTORY_CSV = Path("complaint_history.csv")

# ---------------- Light UI styling ----------------
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

# ---------------- Cached loaders ----------------
@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    if not path.exists():
        return None
    obj = joblib.load(path)

    # Handle dict-based joblib
    if isinstance(obj, dict):
        if "pipeline" in obj:
            return obj["pipeline"]
        if "model" in obj:
            return obj["model"]
        return None

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

    model_name = "all-MiniLM-L6-v2"
    if SBERT_META_PATH.exists():
        meta = joblib.load(SBERT_META_PATH)
        model_name = meta.get("sbert_model_name", model_name)

    return SentenceTransformer(model_name)


pipeline = load_pipeline(PIPELINE_PATH)
bank = load_bank(BANK_PATH)
sbert = load_sbert()
bank_embs = np.load(BANK_EMB_PATH) if BANK_EMB_PATH.exists() else None

# ---------------- Rule-based overrides ----------------
LABEL_KEYWORDS = {
    "delivery": ["delivery", "courier", "parcel", "late delivery"],
    "billing": ["refund", "billing", "charged", "payment"],
    "product": ["damaged", "broken", "wrong product", "defective"],
    "account": ["login", "password", "otp", "account"],
    "technical": ["bug", "error", "crash", "not working"],
}

def rule_override_label(text: str, model_label: str, conf: Optional[float]) -> str:
    t = text.lower()
    scores = {k: 0 for k in LABEL_KEYWORDS}
    for label, keys in LABEL_KEYWORDS.items():
        for k in keys:
            if k in t:
                scores[label] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return model_label
    if conf and conf >= 0.8:
        return model_label
    return best


# ---------------- Core inference ----------------
def get_reply(text: str, sim_threshold: float = 0.65):
    # Retrieval
    if sbert and bank and bank_embs is not None:
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

    if pipeline is None:
        return {"method": "error", "message": "Model not loaded correctly."}

    pred = pipeline.predict([text])[0]
    try:
        conf = float(np.max(pipeline.predict_proba([text])[0]))
    except Exception:
        conf = None

    final_label = rule_override_label(text, pred, conf)

    default_replies = {
        "billing": "We understand your billing concern and will resolve it shortly.",
        "delivery": "We are checking the delivery issue and will update you soon.",
        "product": "We have noted the product issue and will assist you.",
        "account": "We will resolve your account-related issue shortly.",
        "technical": "Our technical team is working on this issue.",
    }

    return {
        "method": "classifier",
        "label": final_label,
        "reply": default_replies.get(final_label, "We will look into this issue."),
        "confidence": conf,
    }


# ---------------- Persistent history ----------------
if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        st.session_state["history"] = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        st.session_state["history"] = []

# ---------------- Sidebar ----------------
mode = st.sidebar.radio("View as", ["User panel", "Admin panel"])

st.sidebar.markdown("---")
st.sidebar.write("Pipeline loaded:", pipeline is not None)
st.sidebar.write("Bank loaded:", bank is not None)
st.sidebar.write("SBERT loaded:", sbert is not None)
st.sidebar.write("Bank embeddings:", bank_embs is not None)

# ---------------- USER PANEL ----------------
if mode == "User panel":
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Complaint Auto-Responder</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Submit your complaint and get an instant response.</div>", unsafe_allow_html=True)

        complaint = st.text_area("Enter the complaint", height=150)
        if st.button("Submit", type="primary"):
            if complaint.strip():
                result = get_reply(complaint.strip())
                if result.get("method") == "error":
                    st.error(result["message"])
                else:
                    record = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "complaint": complaint,
                        "label": result["label"],
                        "method": result["method"],
                        "confidence": result["confidence"],
                        "reply": result["reply"],
                    }
                    st.session_state["history"].append(record)
                    pd.DataFrame([record]).to_csv(
                        HISTORY_CSV,
                        mode="a",
                        header=not HISTORY_CSV.exists(),
                        index=False,
                    )

                    st.markdown("<div class='reply-box'>"
                                f"We received your complaint related to <b>{result['label']}</b>:<br>{result['reply']}"
                                "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ADMIN PANEL ----------------
else:
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)

        if not st.session_state["history"]:
            st.info("No complaints yet.")
        else:
            df = pd.DataFrame(st.session_state["history"])
            st.dataframe(df, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
