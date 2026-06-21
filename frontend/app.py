import requests
import streamlit as st

API_URL = 'http://127.0.0.1:8000/predict'

ACTION_EMOJI = {
    'squat':          '🏋️',
    'pushup':         '💪',
    'situp':          '🤸',
    'pullup':         '🙌',
    'jumping_jacks':  '⭐',
    'jump_rope':      '🪢',
    'bench_press':    '🏋️',
    'clean_and_jerk': '🥇',
}

st.set_page_config(page_title='Fitness Assistant', page_icon='💪', layout='wide')

st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #8b8fa8;
        font-size: 1.1rem;
    }

    .result-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2e3550;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-action {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .result-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2e3550;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label {
        color: #8b8fa8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .section-title {
        color: #8b8fa8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.8rem 0;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
        gap: 0.8rem;
    }
    .prob-label {
        color: #c0c4d6;
        font-size: 0.9rem;
        width: 140px;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        background: #1a1f2e;
        border-radius: 99px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    }
    .prob-pct {
        color: #8b8fa8;
        font-size: 0.85rem;
        width: 42px;
        text-align: right;
        flex-shrink: 0;
    }
    .top-prob-label { color: #ffffff; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💪 Fitness Assistant</h1>
    <p>Upload a workout video — AI identifies the exercise and counts your reps</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: upload left, results right ───────────────────────────────────────
left, right = st.columns([1, 1], gap='large')

with left:
    uploaded = st.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov', 'mkv'],
                                label_visibility='collapsed')
    if uploaded:
        st.video(uploaded)
        analyze = st.button('Analyze', type='primary', use_container_width=True)
    else:
        st.markdown("""
        <div style="border: 2px dashed #2e3550; border-radius: 12px; padding: 3rem;
                    text-align: center; color: #8b8fa8; margin-top: 0.5rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">🎬</div>
            <div>Drop a video here</div>
            <div style="font-size:0.8rem; margin-top:0.3rem;">MP4 · AVI · MOV · MKV</div>
        </div>
        """, unsafe_allow_html=True)
        analyze = False

with right:
    if uploaded and analyze:
        with st.spinner('Analyzing your workout...'):
            uploaded.seek(0)
            response = requests.post(
                API_URL,
                files={'file': (uploaded.name, uploaded, uploaded.type)},
            )

        if response.status_code == 200:
            result     = response.json()
            action     = result['action']
            confidence = result['confidence']
            reps       = result.get('reps', 0)
            probs      = result['probabilities']
            emoji      = ACTION_EMOJI.get(action, '🏃')

            # Result card
            st.markdown(f"""
            <div class="result-card">
                <div class="result-emoji">{emoji}</div>
                <div class="result-action">{action.replace("_", " ").title()}</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.0%}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Reps Counted</div>
                    <div class="metric-value">{reps}</div>
                </div>""", unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="section-title">All predictions</div>', unsafe_allow_html=True)
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            bars_html = ''
            for i, (label, prob) in enumerate(sorted_probs):
                label_class = 'prob-label top-prob-label' if i == 0 else 'prob-label'
                bars_html += f"""
                <div class="prob-row">
                    <span class="{label_class}">{label.replace("_", " ").title()}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{prob*100:.1f}%"></div>
                    </div>
                    <span class="prob-pct">{prob:.0%}</span>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

        else:
            st.error(f'Error: {response.json().get("detail", "Unknown error")}')

    elif not uploaded:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center;
                    height:100%; min-height:300px; color:#8b8fa8; text-align:center;">
            <div>
                <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
                <div style="font-size:1rem;">Results will appear here</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
