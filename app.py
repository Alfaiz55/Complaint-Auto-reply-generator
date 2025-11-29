# app.py – Complaint Auto-Responder with User/Admin panels, rules, CSV persistence, extra info field

import time
from pathlib import Path
from typing import Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# -------------- Page config --------------
st.set_page_config(
    page_title="Complaint Auto-Responder",
    layout="centered",
)

# -------------- Paths --------------
BASE_DIR = Path(__file__).parent

PIPELINE_PATH   = BASE_DIR / "pipeline_calibrated.joblib"
BANK_PATH       = BASE_DIR / "complaint_bank.pkl"
SBERT_META_PATH = BASE_DIR / "sbert_meta.joblib"
BANK_EMB_PATH   = BASE_DIR / "bank_embeddings.npy"      # if missing, retrieval will be skipped
HISTORY_CSV     = BASE_DIR / "complaint_history.csv"    # Path

# -------------- Light UI styling --------------
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
    .hint-text {
        color: #6b7280;
        font-size: 13px;
        margin-top: 4px;
        margin-bottom: 16px;
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

# -------------- Cached loaders --------------
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

# -------------- Rule-based overrides --------------
LABEL_KEYWORDS = {
    "delivery": [
        "delivery boy", "delivery", "courier", "parcel", "package",
        "didn't come my home", "did not come my home", "not delivered",
        "late delivery", "wrong address", "doorstep", "asking extra money",
        "delivery cancelled",
    ],
    "billing": [
        "charged twice", "double charged", "extra charge", "extra amount",
        "refund", "billing", "invoice", "payment issue",
        "deducted from my account", "wrong amount", "transaction failed",
    ],
    "product": [
        "damaged product", "broken", "defective", "wrong colour",
        "wrong color", "wrong size", "quality issue", "product issue",
        "item is not working", "scratches", "wrong product",
        "wrong febric", "different brand",
    ],
    "account": [
        "login", "log in", "password", "reset", "otp", "account",
        "cannot access", "can't access", "blocked", "suspended",
        "coupon not received", "coupon not get"
    ],
    "technical": [
        "app crashes", "crash", "bug", "error", "not loading",
        "server down", "technical issue", "website issue",
        "app is not working", "hangs", "product not show",
        "product price not show",
    ],
}

ALL_LABELS = ["billing", "delivery", "product", "account", "technical"]

def rule_override_label(text: str, model_label: str, conf: Optional[float]) -> str:
    t = text.lower()
    hits = {label: 0 for label in LABEL_KEYWORDS.keys()}
    for label, kws in LABEL_KEYWORDS.items():
        for k in kws:
            if k in t:
                hits[label] += 1
    best = max(hits, key=hits.get)
    if hits[best] == 0:
        return model_label
    if conf is not None and conf >= 0.80:
        return model_label
    return best if best != model_label else model_label

# -------------- Core inference logic --------------
def get_reply(text: str, sim_threshold: float = 0.65):
    # Retrieval
    if sbert is not None and bank is not None and bank_embs is not None:
        q = sbert.encode([text], convert_to_numpy=True)
        sims = cosine_similarity(q, bank_embs)[0]
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        if best_sim >= sim_threshold:
            m = bank[idx]
            return {
                "method": "retrieval",
                "confidence": best_sim,
                "label": m["label"],
                "reply": m["reply"],
                "matched_text": m["text"],
            }
    # Classifier
    if pipeline is None:
        return {"method": "error", "message": "Model not loaded."}

    pred = pipeline.predict([text])[0]
    try:
        probs = pipeline.predict_proba([text])[0]
        conf = float(np.max(probs))
    except Exception:
        conf = None

    final_label = rule_override_label(text, pred, conf)

    default_replies = {
        "billing":   "Thanks for telling us — we understand. We'll check your billing issue and update you as soon as possible.",
        "delivery":  "Thanks for telling us — we understand. We'll investigate the delivery problem and get back to you soon. And ensure this will not happen again.",
        "product":   "Thanks for telling us — we understand. We'll review your product issue and provide a resolution as soon as possible.And ensure this will not happen again.",
        "account":   "Thanks for telling us — we understand. We'll solving your account problem and share an update as soon as possible. Also with proper mail and notification by system",
        "technical": "Thanks for telling us — we understand. Our technical team will check this issue and we'll update you as soon as possible.",
    }

    reply = default_replies.get(
        final_label,
        "Thanks for telling us — we understand. We'll look into this and update you soon.",
    )

    return {
        "method": "classifier",
        "confidence": conf,
        "label": final_label,
        "reply": reply,
        "matched_text": None,
    }

# -------------- Persistent History (CSV) --------------
if "history" not in st.session_state:
    if HISTORY_CSV.exists():
        df_hist = pd.read_csv(HISTORY_CSV)
        st.session_state["history"] = df_hist.to_dict("records")
    else:
        st.session_state["history"] = []

# pending state for 2-step submission (prediction → extra info → save)
if "pending_result" not in st.session_state:
    st.session_state["pending_result"] = None
if "pending_complaint" not in st.session_state:
    st.session_state["pending_complaint"] = ""
if "pending_time" not in st.session_state:
    st.session_state["pending_time"] = ""
if "pending_elapsed" not in st.session_state:
    st.session_state["pending_elapsed"] = 0.0
if "extra_info" not in st.session_state:
    st.session_state["extra_info"] = ""

# -------------- Sidebar --------------
mode = st.sidebar.radio("View as", ["User panel", "Admin panel"], index=0)

st.sidebar.markdown("---")
st.sidebar.write("Pipeline loaded:", pipeline is not None)
st.sidebar.write("Bank loaded:", bank is not None)
st.sidebar.write("SBERT loaded:", sbert is not None)
st.sidebar.write("Bank embeddings:", bank_embs is not None)

# -------------- USER PANEL --------------
if mode == "User panel":
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Complaint Auto-Responder</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Enter your complaint related to our services. "
            "And we give you immediate response and provide the helpful service as soon as possible.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-title'>Enter the complaint</div>", unsafe_allow_html=True)
        complaint = st.text_area(
            "Enter the complaint",
            height=150,
            placeholder="Example: The delivery boy asked me to come to the office to take the product. He did not come to my home for delivery.",
            key="complaint_input",
        )

        submit = st.button("Submit", type="primary")

        # Step 1: prediction
        if submit:
            if not complaint.strip():
                st.warning("Please enter a complaint before submitting.")
            else:
                with st.spinner("Analyzing your complaint and generating reply..."):
                    start = time.time()
                    result = get_reply(complaint.strip())
                    elapsed = time.time() - start

                if result.get("method") == "error":
                    st.error(result.get("message"))
                    st.session_state["pending_result"] = None
                else:
                    # store pending info to complete after extra field
                    st.session_state["pending_result"] = result
                    st.session_state["pending_complaint"] = complaint.strip()
                    st.session_state["pending_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["pending_elapsed"] = elapsed
                    st.session_state["extra_info"] = ""  # reset

        # Step 2: show reply + required extra textbox + save button
        pending = st.session_state.get("pending_result")
        if pending is not None:
            label = pending.get("label", "unknown")
            reply_text = pending.get("reply", "")
            elapsed = st.session_state.get("pending_elapsed", 0.0)

            st.markdown("<div class='section-title'>Suggested response</div>", unsafe_allow_html=True)
            display_text = f"We received your complaint related to **{label}** :-<br>{reply_text}"
            st.markdown(f"<div class='reply-box'>{display_text}</div>", unsafe_allow_html=True)

            meta = []
            if pending.get("method"):
                meta.append(f"Method: `{pending['method']}`")
            if pending.get("confidence") is not None:
                meta.append(f"Confidence: `{pending['confidence']:.2f}`")
            meta.append(f"Generated in {elapsed:.2f}s")

            st.markdown(
                "<div class='meta-text'>" + " • ".join(meta) + "</div>",
                unsafe_allow_html=True,
            )

            # choose label-specific extra field
            if label in ("product", "delivery"):
                extra_label = "Enter your Order ID"
                extra_placeholder = "Example: ORD-123456"
            elif label == "billing":
                extra_label = "Enter your Billing / Transaction ID"
                extra_placeholder = "Example: TXN-987654321"
            elif label == "account":
                extra_label = "Enter your Username or registered email"
                extra_placeholder = "Example: user123 or name@gmail.com"
            elif label == "technical":
                extra_label = "Enter your Device / App details"
                extra_placeholder = "Example: Phone model, Android version, App version"
            else:
                extra_label = "Enter related details"
                extra_placeholder = "Example: useful reference for this complaint"

            st.markdown("<div class='section-title'>Additional required details</div>", unsafe_allow_html=True)
            extra_val = st.text_input(
                extra_label,
                value=st.session_state.get("extra_info", ""),
                placeholder=extra_placeholder,
                key="extra_info_input",
            )

            # Keep in session so we can validate on button click
            st.session_state["extra_info"] = extra_val

            save_btn = st.button("Save complaint")

            if save_btn:
                if not extra_val.strip():
                    st.warning("Please fill the required details before saving.")
                else:
                    # build record and persist
                    record = {
                        "time": st.session_state.get("pending_time"),
                        "complaint": st.session_state.get("pending_complaint"),
                        "label": label,
                        "method": pending.get("method"),
                        "confidence": pending.get("confidence"),
                        "reply": reply_text,
                        "extra_info": extra_val.strip(),
                    }

                    # Save to session
                    st.session_state["history"].append(record)

                    # Save to CSV
                    df_new = pd.DataFrame([record])
                    if HISTORY_CSV.exists():
                        df_new.to_csv(HISTORY_CSV, mode="a", index=False, header=False)
                    else:
                        df_new.to_csv(HISTORY_CSV, mode="w", index=False, header=True)

                    st.success(
                        "Your complaint and additional details have been saved. "
                        "We appreciate your patience and will get back to you shortly."
                    )

                    # clear pending state
                    st.session_state["pending_result"] = None
                    st.session_state["pending_complaint"] = ""
                    st.session_state["pending_time"] = ""
                    st.session_state["pending_elapsed"] = 0.0
                    st.session_state["extra_info"] = ""
                    st.session_state["extra_info_input"] = ""

        st.markdown("</div>", unsafe_allow_html=True)

# -------------- ADMIN PANEL --------------
else:
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Admin Panel</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Statistics and history of all complaints processed by the system.</div>",
            unsafe_allow_html=True,
        )

        history = st.session_state["history"]

        if not history:
            st.info("No complaints processed yet.")
        else:
            df = pd.DataFrame(history)

            # ensure extra_info column exists for old CSVs
            if "extra_info" not in df.columns:
                df["extra_info"] = ""

            st.markdown("<div class='section-title'>Complaint Dashboard</div>", unsafe_allow_html=True)
            st.write(f"Total complaints processed: **{len(df)}**")

            # build full counts including zero for missing labels
            counts = df["label"].value_counts()
            label_counts = pd.DataFrame({"label": ALL_LABELS})
            label_counts["count"] = label_counts["label"].map(counts).fillna(0).astype(int)
            total = label_counts["count"].sum()
            if total > 0:
                label_counts["percentage"] = (label_counts["count"] / total * 100).round(2)
            else:
                label_counts["percentage"] = 0.0

            # Show all labels table (index starting from 1)
            label_counts_display = label_counts.copy()
            label_counts_display.index = label_counts_display.index + 1
            st.dataframe(label_counts_display, use_container_width=True)

            # Pie + labels list
            st.markdown("### Complaint categories & distribution")

            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("**All complaint categories:**")
                for lbl in ALL_LABELS:
                    st.markdown(f"- {lbl.capitalize()}")

            with col_right:
                pie_data = label_counts[label_counts["count"] > 0]
                if len(pie_data) > 0:
                    base = (
                        alt.Chart(pie_data)
                        .encode(
                            theta=alt.Theta("count:Q", stack=True),
                            color=alt.Color(
                                "label:N",
                                legend=alt.Legend(title="Labels"),
                            ),
                            tooltip=[
                                alt.Tooltip("label:N", title="Label"),
                                alt.Tooltip("count:Q", title="Count"),
                                alt.Tooltip(
                                    "percentage:Q", title="Percentage", format=".2f"
                                ),
                            ],
                        )
                    )

                    pie = base.mark_arc()
                    text = base.mark_text(radius=110).encode(
                        text=alt.Text("percentage:Q", format=".0f")
                    )

                    chart = (pie + text).properties(height=250)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No data yet for pie chart.")

            st.markdown("<div class='section-title'>Complaint History</div>", unsafe_allow_html=True)

            # History table with index starting from 1, including extra_info
            hist_df = df[["time", "complaint", "label", "method", "confidence", "extra_info"]].copy()
            hist_df.index = hist_df.index + 1
            st.dataframe(hist_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
