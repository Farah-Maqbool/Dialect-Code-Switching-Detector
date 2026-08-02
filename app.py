import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

st.set_page_config(
    page_title="Dialect & Code-Switching Detector",
    # page_icon="🗣️",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f7f8fc; }
.title-block {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 36px 32px 28px 32px;
    margin-bottom: 28px; color: white;
}
.title-block h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.3px; }
.title-block p  { font-size: 0.92rem; opacity: 0.7; margin: 0; }
.token-container {
    display: flex; flex-wrap: wrap; gap: 10px; padding: 20px;
    background: white; border-radius: 12px; border: 1px solid #e8e8f0; margin: 16px 0;
}
.token-chip { display: flex; flex-direction: column; align-items: center; gap: 5px; }
.token-word-urdu {
    background: #dbeafe; color: #1e40af; padding: 7px 13px;
    border-radius: 8px; font-size: 1rem; font-weight: 600; border: 1.5px solid #bfdbfe;
}
.token-word-english {
    background: #dcfce7; color: #166534; padding: 7px 13px;
    border-radius: 8px; font-size: 1rem; font-weight: 600; border: 1.5px solid #bbf7d0;
}
.token-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.8px; color: #888; text-transform: uppercase; }
.switch-arrow { color: #f97316; font-size: 1.1rem; align-self: center; margin-top: -6px; }
.dialect-card { border-radius: 12px; padding: 20px 24px; margin: 8px 0; }
.dialect-codemix   { background: #fff7ed; border: 1.5px solid #fed7aa; }
.dialect-romanurdu { background: #eff6ff; border: 1.5px solid #bfdbfe; }
.dialect-english   { background: #f0fdf4; border: 1.5px solid #bbf7d0; }
.dialect-label-text { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.3px; }
.codemix-color   { color: #c2410c; }
.romanurdu-color { color: #1d4ed8; }
.english-color   { color: #15803d; }
.dialect-desc { font-size: 0.85rem; color: #666; margin-top: 4px; }
.switch-info {
    background: #fafafa; border: 1px solid #e8e8f0; border-radius: 10px;
    padding: 14px 18px; margin-top: 12px; font-size: 0.88rem; color: #444;
}
.legend-row { display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.8rem; color: #666; padding: 10px 0 4px 0; }
.legend-item { display: flex; align-items: center; gap: 7px; }
.legend-dot-urdu    { width:12px; height:12px; border-radius:4px; background:#dbeafe; border:1.5px solid #1e40af; display:inline-block; }
.legend-dot-english { width:12px; height:12px; border-radius:4px; background:#dcfce7; border:1.5px solid #166534; display:inline-block; }
.legend-dot-switch  { width:12px; height:12px; border-radius:4px; background:#f97316; display:inline-block; }
.stat-row { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
.stat-box {
    flex: 1; min-width: 100px; background: white; border: 1px solid #e8e8f0;
    border-radius: 10px; padding: 12px 16px; text-align: center;
}
.stat-number { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
.stat-label  { font-size: 0.75rem; color: #888; margin-top: 2px; }
.section-title { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #888; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    MODEL_PATH = "./lang_detect_model_final"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH)
    model      = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def predict_token_tags(sentence, tokenizer, model):
    id2label = model.config.id2label
    words    = sentence.strip().split()
    if not words:
        return []
    encoding = tokenizer(
        words, is_split_into_words=True,
        return_tensors='pt', truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs     = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = encoding.word_ids()
    results, seen = [], set()
    for idx, wid in enumerate(word_ids):
        if wid is not None and wid not in seen:
            results.append((words[wid], id2label[predictions[idx]]))
            seen.add(wid)
    return results

def classify_dialect(token_tags, eng_threshold=0.70, urdu_threshold=0.80):
    if not token_tags:
        return 'UNKNOWN', 0, 0
    tags      = [t for _, t in token_tags]
    total     = len(tags)
    eng_frac  = tags.count('ENGLISH') / total
    urdu_frac = tags.count('URDU')    / total
    if eng_frac >= eng_threshold:
        return 'ENGLISH', round(eng_frac * 100), round(urdu_frac * 100)
    elif urdu_frac >= urdu_threshold:
        return 'ROMAN_URDU', round(eng_frac * 100), round(urdu_frac * 100)
    else:
        return 'CODE_MIX', round(eng_frac * 100), round(urdu_frac * 100)

def detect_switches(token_tags):
    switches = []
    for i in range(1, len(token_tags)):
        if token_tags[i][1] != token_tags[i-1][1]:
            switches.append(i)
    return switches

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🗣️ Dialect & Code-Switching Detector</h1>
    
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-title">Try an example</p>', unsafe_allow_html=True)
examples = [
    "yaar I am so tired aaj",
    "bro this project bohot zyada kaam hai",
    "I have no idea what happened in class today",
    "aaj ka lecture honestly boring tha yaar",
    "yr mujy nhi pta kia ho rha hai",
    "hum log basically wait kar rahe hain result ka",
]
cols = st.columns(3)
selected_example = None
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, use_container_width=True, key=f"ex_{i}"):
        selected_example = ex

user_input = st.text_area(
    "Enter any Pakistani sentence",
    value=selected_example if selected_example else "",
    placeholder="Type Roman Urdu, English, or mixed text here...",
    height=100, label_visibility="collapsed"
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

if analyze_btn and user_input.strip():
    with st.spinner("Running mBERT model..."):
        try:
            tokenizer, model = load_model()
            token_tags = predict_token_tags(user_input.strip(), tokenizer, model)
            dialect, eng_pct, urdu_pct = classify_dialect(token_tags)
            switches = detect_switches(token_tags)
        except Exception as e:
            st.error(f"Model error: {e}. Make sure lang_detect_model_final folder is in the same directory.")
            st.stop()

    st.markdown("---")

    dialect_class   = {'CODE_MIX': 'codemix', 'ROMAN_URDU': 'romanurdu', 'ENGLISH': 'english'}.get(dialect, 'codemix')
    dialect_desc    = {
        'CODE_MIX'   : 'This sentence mixes Urdu and English — code-switching detected',
        'ROMAN_URDU' : 'This sentence is predominantly Roman Urdu',
        'ENGLISH'    : 'This sentence is predominantly English',
    }.get(dialect, '')
    dialect_display = dialect.replace('_', ' ')

    st.markdown('<p class="section-title">Dialect</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="dialect-card dialect-{dialect_class}">
        <div class="dialect-label-text {dialect_class}-color">{dialect_display}</div>
        <div class="dialect-desc">{dialect_desc}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-number">{len(token_tags)}</div><div class="stat-label">Total Tokens</div></div>
        <div class="stat-box"><div class="stat-number" style="color:#166534">{eng_pct}%</div><div class="stat-label">English</div></div>
        <div class="stat-box"><div class="stat-number" style="color:#1e40af">{urdu_pct}%</div><div class="stat-label">Urdu</div></div>
        <div class="stat-box"><div class="stat-number" style="color:#f97316">{len(switches)}</div><div class="stat-label">Switch Points</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title" style="margin-top:20px">Token-level language tags</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend-row">
        <div class="legend-item"><span class="legend-dot-urdu"></span> URDU token</div>
        <div class="legend-item"><span class="legend-dot-english"></span> ENGLISH token</div>
        <div class="legend-item"><span class="legend-dot-switch"></span> Switch point</div>
    </div>""", unsafe_allow_html=True)

    token_html = '<div class="token-container">'
    for i, (word, tag) in enumerate(token_tags):
        css_class   = 'token-word-urdu' if tag == 'URDU' else 'token-word-english'
        token_html += f'<div class="token-chip"><div class="{css_class}">{word}</div><div class="token-label">{tag}</div></div>'
        if i in switches:
            token_html += '<div class="switch-arrow">⇄</div>'
    token_html += '</div>'
    st.markdown(token_html, unsafe_allow_html=True)

    if switches:
        switch_words = [f"'{token_tags[i-1][0]}' ({token_tags[i-1][1]}) → '{token_tags[i][0]}' ({token_tags[i][1]})" for i in switches]
        st.markdown(f"""
        <div class="switch-info">
            <b>Code-switching detected at {len(switches)} point(s):</b><br>
            <span style="color:#888; font-size:0.85rem">{' &nbsp;|&nbsp; '.join(switch_words)}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="switch-info">
            <b>No code-switching detected</b> — sentence uses a single language throughout.
        </div>""", unsafe_allow_html=True)

elif analyze_btn and not user_input.strip():
    st.warning("Please enter a sentence first.")