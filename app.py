import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
import time

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Note Classifier",
    page_icon="🏥",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c0f;
    color: #e8f4f8;
}

.stApp {
    background: linear-gradient(135deg, #080c0f 0%, #0d1418 100%);
}

/* Remove Streamlit default padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 780px;
}

/* ── Header ── */
.hero-badge {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 100px;
    padding: 6px 18px;
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #00d4ff;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ffffff 30%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 6px;
}

.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #5a7a8a;
    letter-spacing: 1px;
    margin-bottom: 32px;
}

/* ── Card ── */
.glass-card {
    background: #111920;
    border: 1px solid #1e2e38;
    border-radius: 16px;
    padding: 28px 32px;
    position: relative;
    margin-bottom: 24px;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff55, transparent);
    border-radius: 16px 16px 0 0;
}

/* ── Section label ── */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #00d4ff;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* ── Result prediction ── */
.pred-card {
    background: #0d1418;
    border: 1px solid #1e2e38;
    border-radius: 12px;
    padding: 22px 26px;
    position: relative;
    overflow: hidden;
}

.pred-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #00ff9d, #00d4ff);
}

.pred-label-small {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #5a7a8a;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.pred-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00ff9d;
    letter-spacing: -0.5px;
}

.conf-value {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00ff9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Bar styling ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.bar-label {
    width: 160px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #8c9ca8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.bar-track {
    flex: 1;
    height: 6px;
    background: #0d1418;
    border: 1px solid #1e2e38;
    border-radius: 100px;
    overflow: hidden;
}

.bar-fill-top {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00ff9d, #00d4ff);
    box-shadow: 0 0 8px #00d4ff55;
}

.bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #1e2e38, #2a404f);
}

.bar-pct {
    width: 44px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #5a7a8a;
    text-align: right;
}

/* ── Streamlit widget overrides ── */
textarea {
    background: #0d1418 !important;
    border: 1px solid #1e2e38 !important;
    border-radius: 10px !important;
    color: #e8f4f8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 14px !important;
}

textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
}

.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.14), rgba(0,255,157,0.08)) !important;
    border: 1px solid rgba(0,212,255,0.4) !important;
    border-radius: 10px !important;
    color: #00d4ff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(0,255,157,0.14)) !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.2) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* pill row */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 20px 0;
}

.pill {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 11px;
    font-family: 'Share Tech Mono', monospace;
    color: #5a7a8a;
    cursor: pointer;
    white-space: nowrap;
}

.stAlert {
    border-radius: 10px !important;
}

/* Footer */
.footer {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #2a404f;
    letter-spacing: 1px;
    margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── Model Configuration ───────────────────────────────────────────────────────
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

SPECIALTIES = {
    "Cardiology": [
        "chest pain heart attack myocardial infarction coronary artery disease arrhythmia",
        "palpitations angina shortness of breath left arm pain cardiac catheterization",
        "atrial fibrillation hypertension blood pressure ECG echocardiogram",
        "tachycardia bradycardia cardiac arrest ventricular fibrillation bypass surgery",
    ],
    "Neurology": [
        "headache migraine seizure stroke brain nerve cognitive",
        "dizziness tremor Parkinson multiple sclerosis epilepsy memory loss",
        "MRI brain scan neurological assessment cerebral infarction",
        "numbness tingling weakness facial droop aphasia neuropathy",
    ],
    "Orthopedics": [
        "fracture bone joint surgery arthritis tendon ligament",
        "back pain spinal stenosis knee hip shoulder orthopedic",
        "osteoporosis scoliosis ACL meniscus cartilage replacement",
        "physical therapy musculoskeletal pain range of motion",
    ],
    "Pulmonology": [
        "cough breathing asthma bronchitis pneumonia lung infection",
        "wheezing oxygen saturation COPD pulmonary respiratory",
        "sputum dyspnea spirometry inhaler nebulizer pleural effusion",
        "shortness of breath chest tightness sleep apnea tuberculosis",
    ],
    "Endocrinology": [
        "diabetes insulin glucose blood sugar thyroid hormone pancreas",
        "hyperglycemia hypoglycemia HbA1c endocrine metabolism cortisol",
        "hypothyroidism hyperthyroidism adrenal gland obesity",
        "polycystic ovary syndrome PCOS metabolic syndrome weight gain",
    ],
    "Gastroenterology": [
        "stomach pain diarrhea constipation nausea vomiting abdominal",
        "ulcer acid reflux GERD gastritis bowel irritable colon",
        "liver hepatitis cirrhosis endoscopy colonoscopy biopsy",
        "Crohn's disease colitis pancreatitis jaundice indigestion",
    ],
    "Dermatology": [
        "rash acne eczema psoriasis skin lesion biopsy",
        "melanoma dermatitis hives urticaria photosensitivity",
        "topical steroids skin infection fungal bacterial treatment",
        "mole itching wound cellulitis warts seborrheic keratosis",
    ],
    "Psychiatry": [
        "depression anxiety disorder mood mental health psychiatry",
        "schizophrenia bipolar PTSD panic attack phobia obsessive",
        "antidepressant SSRI psychotherapy cognitive behavioral therapy",
        "hallucinations delusions suicidal ideation emotional disturbance",
    ],
    "Oncology": [
        "cancer tumor malignant chemotherapy radiation biopsy",
        "lymphoma leukemia breast lung colorectal prostate cancer",
        "metastasis oncologist staging prognosis remission",
        "immunotherapy targeted therapy palliative care neoplasm",
    ],
    "Nephrology": [
        "kidney renal failure dialysis creatinine urine proteinuria",
        "glomerulonephritis hypertension electrolyte nephropathy",
        "urinary tract infection hematuria acute chronic kidney disease",
        "transplant hydronephrosis polycystic kidney biopsy",
    ],
}

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def build_label_embeddings(_tokenizer, _model):
    def mean_pool(token_embs, attn_mask):
        mask = attn_mask.unsqueeze(-1).expand(token_embs.size()).float()
        return torch.sum(token_embs * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def get_emb(text):
        enc = _tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = _model(**enc)
        return mean_pool(out.last_hidden_state, enc["attention_mask"])

    label_names = list(SPECIALTIES.keys())
    label_embs = []
    for label, phrases in SPECIALTIES.items():
        phrase_embs = torch.cat([get_emb(p) for p in phrases], dim=0)
        label_embs.append(phrase_embs.mean(dim=0, keepdim=True))

    label_embs = torch.cat(label_embs, dim=0)
    return label_names, F.normalize(label_embs, p=2, dim=-1)

def predict(text, tokenizer, model, label_names, label_embeddings):
    """Zero-shot cosine-similarity classification."""
    enc = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    query_emb = torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    query_emb = F.normalize(query_emb, p=2, dim=-1)

    sims = (query_emb @ label_embeddings.T).squeeze(0)
    probs = F.softmax(sims * 12.0, dim=0)

    best_idx = int(torch.argmax(probs).item())
    all_scores = {label_names[i]: round(float(probs[i].item()), 4) for i in range(len(label_names))}
    return label_names[best_idx], round(float(probs[best_idx].item()), 4), all_scores

# ── Header Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">● Bio_ClinicalBERT · Zero-Shot Inference</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Medical Note Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered · Clinical Text Analysis · Instant Results</div>', unsafe_allow_html=True)

# ── Load Model Indicator ───────────────────────────────────────────────────────
with st.spinner("Loading ClinicalBERT model... (first load may take a moment)"):
    tokenizer, model = load_model()
    label_names, label_embeddings = build_label_embeddings(tokenizer, model)

# ── Input Card ─────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="section-label">Clinical Note Input</div>', unsafe_allow_html=True)

note_text = st.text_area(
    label="",
    height=170,
    placeholder="Enter a clinical note...\ne.g. Patient reports severe chest pain radiating to the left arm and shortness of breath.",
    label_visibility="collapsed",
    key="note_input"
)

# Quick-fill examples
st.markdown("""
<div class="pill-row">
  <span class="pill">Chest pain, shortness of breath</span>
  <span class="pill">Partial seizure, MRI ordered</span>
  <span class="pill">Elevated HbA1c, insulin adjusted</span>
  <span class="pill">ACL tear, knee surgery advised</span>
  <span class="pill">Bipolar disorder, lithium prescribed</span>
  <span class="pill">Chronic cough, COPD spirometry</span>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

run_btn = st.button("🔮  Run Classification", use_container_width=True)

# ── Prediction Logic ────────────────────────────────────────────────────────────
if run_btn:
    if not note_text.strip():
        st.warning("Please enter a clinical note to classify.")
    else:
        with st.spinner("Classifying..."):
            time.sleep(0.4)
            predicted_label, confidence, all_scores = predict(
                note_text.strip(), tokenizer, model, label_names, label_embeddings
            )

        # ── Result Display ─────────────────────────────────────────────────
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div class="glass-card pred-card">
                <div class="pred-label-small">Predicted Specialty</div>
                <div class="pred-value">🏥 {predicted_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="pred-label-small">Confidence</div>
                <div class="conf-value">{confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Score Bars ─────────────────────────────────────────────────────
        st.markdown('<div class="glass-card"><div class="section-label">All Category Scores</div>', unsafe_allow_html=True)

        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        max_score = sorted_scores[0][1] if sorted_scores else 1

        bars_html = ""
        for label, score in sorted_scores:
            pct = (score / max_score) * 100
            is_top = label == predicted_label
            bar_class = "bar-fill-top" if is_top else "bar-fill"
            name_style = "color:#00ff9d;" if is_top else ""
            bars_html += f"""
            <div class="bar-row">
              <span class="bar-label" style="{name_style}">{label}</span>
              <div class="bar-track"><div class="{bar_class}" style="width:{pct:.1f}%"></div></div>
              <span class="bar-pct" style="{name_style}">{score*100:.1f}%</span>
            </div>"""

        st.markdown(bars_html + '</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Medical Note Classifier · Bio_ClinicalBERT · Inference Only · No Training Required</div>', unsafe_allow_html=True)
