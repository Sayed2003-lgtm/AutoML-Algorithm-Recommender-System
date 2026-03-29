import streamlit as st
import pandas as pd
import numpy as np
import time
from recommender import analyze_dataset, recommend_algorithm

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .big-title { font-family: 'Syne', sans-serif !important; }

/* Dark background */
.stApp { background: #0a0f1e; color: #e8eaf0; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #0a0f1e 60%, #0d2a1e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 50px 40px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(0,200,120,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(0,200,120,0.15);
    border: 1px solid rgba(0,200,120,0.4);
    color: #00c878;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 18px;
    text-transform: uppercase;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
    margin: 0 0 16px;
}
.hero-title span { color: #00c878; }
.hero-sub {
    color: #8892a4;
    font-size: 1.05rem;
    font-weight: 300;
    line-height: 1.7;
    max-width: 560px;
}

/* Upload card */
.upload-card {
    background: #111827;
    border: 2px dashed #1e3a5f;
    border-radius: 16px;
    padding: 36px;
    text-align: center;
    transition: border-color .3s;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #0d2a1e, #0a1a0f);
    border: 1px solid #00c87840;
    border-radius: 16px;
    padding: 32px;
    margin: 24px 0;
}
.algo-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #00c878;
}
.confidence-bar-wrap {
    background: #1a2535;
    border-radius: 999px;
    height: 10px;
    margin: 10px 0 4px;
    overflow: hidden;
}
.confidence-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #00c878, #00e5ff);
    transition: width 1s ease;
}

/* Metric cards */
.metric-grid { display: flex; gap: 16px; flex-wrap: wrap; margin: 24px 0; }
.metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    flex: 1; min-width: 140px;
}
.metric-label { color: #8892a4; font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { color: #ffffff; font-size: 1.4rem; font-weight: 600; font-family: 'Syne', sans-serif; }
.metric-value.green { color: #00c878; }

/* Algorithm row */
.algo-row {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.algo-rank { color: #8892a4; font-size: 13px; width: 24px; }
.algo-label { color: #e8eaf0; flex: 1; font-size: 14px; font-weight: 500; }
.algo-score { color: #00c878; font-weight: 600; font-size: 14px; }
.algo-bar-bg { background: #1a2535; border-radius: 999px; height: 6px; width: 100px; overflow: hidden; }
.algo-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #00c878, #00e5ff); }

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .5px;
}
.badge-green { background: rgba(0,200,120,.15); color: #00c878; border: 1px solid #00c87840; }
.badge-blue  { background: rgba(0,180,255,.15); color: #00b4ff; border: 1px solid #00b4ff40; }

/* Section title */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 28px 0 14px;
    display: flex; align-items: center; gap: 8px;
}

/* Insight box */
.insight-box {
    background: #111827;
    border-left: 3px solid #00c878;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #b0bac8;
    font-size: 14px;
    line-height: 1.6;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">⚡ Hybrid AutoML System</div>
  <div class="hero-title">Algorithm<br><span>Recommender</span></div>
  <div class="hero-sub">
    Upload any CSV dataset and instantly discover the best machine learning algorithm —
    powered by meta-learning, similarity search, and smart weighting.
    No training required.
  </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown('<div class="section-title">📂 Upload Your Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

    target_col = None
    task_type  = None
    df         = None

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

        cols = df.columns.tolist()
        target_col = st.selectbox("🎯 Select Target Column", cols, index=len(cols)-1)

        if target_col:
            nunique = df[target_col].nunique()
            if nunique <= 20 and df[target_col].dtype == object or nunique <= 10:
                task_type = "Classification"
            else:
                task_type = "Regression"
            st.info(f"Detected task type: **{task_type}** ({nunique} unique values in target)")

with col2:
    st.markdown('<div class="section-title">💡 How it works</div>', unsafe_allow_html=True)
    steps = [
        ("🔍", "Analyze", "Dataset shape, types, balance, correlations"),
        ("🧠", "Meta-Learn", "Meta-learning model picks best algorithm"),
        ("📐", "Similarity", "Compares with 50+ reference datasets via FAISS"),
        ("⚖️", "Weighting", "Smart ensemble gives final ranked decision"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="algo-row">
          <span style="font-size:20px">{icon}</span>
          <div>
            <div style="color:#ffffff;font-weight:600;font-size:14px">{title}</div>
            <div style="color:#8892a4;font-size:12px">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── ANALYZE BUTTON ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if df is not None and target_col:
    if st.button("🚀 Analyze & Recommend", use_container_width=True, type="primary"):
        with st.spinner("Analyzing dataset patterns..."):
            progress = st.progress(0, text="Extracting meta-features...")
            time.sleep(0.4)
            stats = analyze_dataset(df, target_col, task_type)
            progress.progress(35, text="Running meta-learning model...")
            time.sleep(0.4)
            progress.progress(65, text="Similarity search across reference datasets...")
            time.sleep(0.4)
            results = recommend_algorithm(stats, task_type)
            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()

        # ── DATASET STATS ──
        st.markdown('<div class="section-title">📊 Dataset Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Rows</div>
            <div class="metric-value">{stats['n_samples']:,}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{stats['n_features']}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Numeric</div>
            <div class="metric-value green">{stats['n_numeric']}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Categorical</div>
            <div class="metric-value">{stats['n_categorical']}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Missing %</div>
            <div class="metric-value {'green' if stats['missing_pct'] < 5 else ''}">{stats['missing_pct']:.1f}%</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Task</div>
            <div class="metric-value green">{task_type}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── TOP RECOMMENDATION ──
        top = results[0]
        st.markdown(f"""
        <div class="result-card">
          <div style="color:#8892a4;font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
            ⭐ Top Recommendation
          </div>
          <div class="algo-name">{top['algorithm']}</div>
          <div style="color:#8892a4;font-size:14px;margin:6px 0 16px">{top['reason']}</div>
          <div style="color:#8892a4;font-size:12px;margin-bottom:4px">Confidence Score</div>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar" style="width:{top['confidence']}%"></div>
          </div>
          <div style="color:#00c878;font-weight:700;font-size:1.1rem">{top['confidence']}%</div>
          <br>
          <span class="badge badge-green">✓ Best for your dataset</span>
          &nbsp;
          <span class="badge badge-blue">{task_type}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── ALL RANKINGS ──
        st.markdown('<div class="section-title">🏆 All Algorithm Rankings</div>', unsafe_allow_html=True)
        for i, r in enumerate(results):
            bar_w = r['confidence']
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
            st.markdown(f"""
            <div class="algo-row">
              <span class="algo-rank">{medal}</span>
              <span class="algo-label">{r['algorithm']}</span>
              <div class="algo-bar-bg"><div class="algo-bar-fill" style="width:{bar_w}%"></div></div>
              <span class="algo-score">{r['confidence']}%</span>
            </div>
            """, unsafe_allow_html=True)

        # ── INSIGHTS ──
        st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)
        for insight in stats["insights"]:
            st.markdown(f'<div class="insight-box">• {insight}</div>', unsafe_allow_html=True)

        # ── RAW STATS EXPANDER ──
        with st.expander("🔬 View Raw Meta-Features"):
            meta_df = pd.DataFrame([{
                "Metric": k.replace("_", " ").title(),
                "Value": str(round(v, 4)) if isinstance(v, float) else str(v)
            } for k, v in stats.items() if k != "insights"])
            st.dataframe(meta_df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:40px;color:#8892a4;">
      <div style="font-size:48px;margin-bottom:12px">📂</div>
      <div style="font-size:1rem;">Upload a CSV file above to get started</div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1e3a5f;margin-top:48px"/>
<div style="text-align:center;color:#4a5568;font-size:13px;padding:16px">
  Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp;
  <b style="color:#00c878">Hybrid AutoML Algorithm Recommender</b>
</div>
""", unsafe_allow_html=True)
