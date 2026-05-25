import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# ── Page config — has to be the very first streamlit call ─────────────────────
st.set_page_config(
    page_title="CancerSense · Breast Cancer Classifier",
    page_icon="🎗️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── All the CSS styles in one block ───────────────────────────────────────────
# Using Google Fonts for a cleaner look.
# Color scheme: teal/emerald — feels more calm and medical than neon.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@500;600;700;800&display=swap');

:root {
    --teal:       #6366f1;
    --teal-light: #a5b4fc;
    --emerald:    #22d3ee;
    --rose:       #f43f5e;
    --amber:      #f59e0b;
    --bg:         #080b14;
    --bg-card:    #0d1120;
    --bg-card2:   #111827;
    --border:     rgba(99,102,241,0.18);
    --text:       #e8eaf6;
    --muted:      #4b5563;
}

#MainMenu, footer, .stDeployButton { visibility: hidden; display: none; }

html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 290px !important;
    max-width: 290px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 1.2rem !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: -290px !important;
    visibility: hidden !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
    flex-direction: column !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.1) !important;
}
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span {
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    margin: 0 !important;
    white-space: nowrap !important;
}

/* predict button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--teal), var(--emerald)) !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}

/* slider thumb */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--teal-light) !important;
    box-shadow: 0 0 8px var(--teal-light) !important;
}

hr { border-color: var(--border) !important; }

.card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    color: var(--teal-light);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.stat-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(13,148,136,0.08);
}
.stat-icon { font-size: 1.05rem; }
.stat-label { font-size: 0.67rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value { font-size: 0.82rem; color: var(--teal-light); font-weight: 600; }

.result-malignant {
    background: linear-gradient(135deg, rgba(244,63,94,0.13), rgba(244,63,94,0.05));
    border: 1px solid rgba(244,63,94,0.4);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.result-benign {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(45,212,191,0.06));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin: 8px 0 4px;
}
.result-malignant .result-title { color: #f43f5e; }
.result-benign   .result-title { color: #10b981; }
.result-conf { font-size: 0.87rem; color: var(--text); opacity: 0.8; margin-top: 4px; }
.result-note { font-size: 0.75rem; color: var(--muted); margin-top: 10px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── Session state — keeps track of which page we're on ────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Classify'


# ── Load dataset defaults once and cache them ─────────────────────────────────
@st.cache_data
def get_defaults():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # use column means as default slider values
    means = df.mean().to_dict()
    median_texture = df['mean texture'].median()
    return means, median_texture


# ── Load the trained model once and cache it ──────────────────────────────────
@st.cache_resource
def load_model():
    # model.pkl should be in the same directory as this script
    model = joblib.load('best_breast_cancer_model.pkl')
    return model


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:28px 0 20px;'>
        <div style='font-family:"Syne",sans-serif; font-size:1.1rem;
                    font-weight:800; color:#a5b4fc; letter-spacing:0.14em; margin-top:10px;'>
            CANCER<span style='color:#22d3ee;'>SENSE</span>
        </div>
        <div style='font-size:0.58rem; color:#4b5563; letter-spacing:0.22em;
                    text-transform:uppercase; margin-top:5px;'>
            ML Diagnostic Tool
        </div>
        <div style='width:45px; height:1px;
                    background:linear-gradient(90deg,transparent,#6366f1,transparent);
                    margin:14px auto 0;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-bottom:8px; font-size:0.6rem; color:#5a7a76;"
        "letter-spacing:0.2em; text-transform:uppercase;'>Navigation</div>",
        unsafe_allow_html=True
    )

    pages = ['About Us', 'Classify', 'About the Study']
    selected = st.radio('nav', pages, index=pages.index(st.session_state.page), label_visibility='hidden')

    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

    st.markdown("""
    <div style='margin-top:3rem; border-top:1px solid rgba(99,102,241,0.1);
                padding-top:1rem; text-align:center; font-size:0.58rem;
                color:#4b5563; letter-spacing:0.12em; text-transform:uppercase;'>
        &copy; 2026 Haider Haroon
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'Classify':

    st.markdown("""
    <div style='text-align:center; padding:2rem 0 0.5rem;'>
        <div style='font-family:"Syne",sans-serif; font-size:1.75rem; font-weight:800;
                    background:linear-gradient(90deg,#a5b4fc,#22d3ee,#6366f1);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    letter-spacing:0.04em; margin-top:10px;'>
            Breast Cancer Classifier
        </div>
        <div style='color:#4b5563; font-size:0.83rem; margin-top:8px;
                    max-width:460px; margin-inline:auto; line-height:1.65;'>
            Adjust the tumor measurements using the sliders below.
            The model will predict whether the tumor is
            <span style='color:#22d3ee; font-weight:600;'>Benign</span> or
            <span style='color:#f43f5e; font-weight:600;'>Malignant</span>.
        </div>
        <div style='width:60px; height:2px;
                    background:linear-gradient(90deg,transparent,#6366f1,transparent);
                    margin:18px auto 30px;'></div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    means, median_texture = get_defaults()

    # ── Input sliders — split into two columns ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        mean_radius     = st.slider('Mean Radius',     6.0,   30.0,    14.0,  step=0.1)
        mean_perimeter  = st.slider('Mean Perimeter',  43.0,  190.0,   91.0,  step=0.5)
        mean_smoothness = st.slider('Mean Smoothness', 0.05,  0.17,    0.096, step=0.001, format='%.3f')

    with col2:
        mean_texture = st.slider('Mean Texture', 9.0,   40.0,  19.0, step=0.1)
        mean_area    = st.slider('Mean Area',    143.0, 2501.0, 654.0, step=1.0)

    # ── Build input dataframe ──────────────────────────────────────────────────
    # start from dataset means, then override with slider values
    user_input = means.copy()
    user_input['mean radius']     = mean_radius
    user_input['mean texture']    = mean_texture
    user_input['mean perimeter']  = mean_perimeter
    user_input['mean area']       = mean_area
    user_input['mean smoothness'] = mean_smoothness

    # these need to match exactly what was used during training
    user_input['tumor_size_category'] = (
        'Small' if mean_radius <= 12 else 'Medium' if mean_radius <= 18 else 'Large'
    )
    user_input['texture_type'] = 'Rough' if mean_texture > median_texture else 'Smooth'

    input_df = pd.DataFrame(user_input, index=[0])

    # ── Predict button ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        predict_clicked = st.button('🔬 Run Diagnosis', type='primary', use_container_width=True)

    if predict_clicked:
        with st.spinner('Analyzing measurements...'):
            prediction = model.predict(input_df)

            # not all models support predict_proba, so handle that gracefully
            try:
                proba    = model.predict_proba(input_df)[0]
                conf_str = f"Model confidence: <strong>{np.max(proba) * 100:.1f}%</strong>"
            except AttributeError:
                conf_str = ''

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # 0 = malignant, 1 = benign
        if prediction[0] == 0:
            st.markdown(f"""
            <div class='result-malignant'>
                <div style='font-size:2.2rem;'>🚨</div>
                <div class='result-title'>MALIGNANT</div>
                <div class='result-conf'>{conf_str}</div>
                <div class='result-note'>
                    For informational purposes only.<br>
                    Please consult a qualified medical professional immediately.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-benign'>
                <div style='font-size:2.2rem;'>✅</div>
                <div class='result-title'>BENIGN</div>
                <div class='result-conf'>{conf_str}</div>
                <div class='result-note'>
                    Measurements suggest benign characteristics.<br>
                    Regular check-ups are still recommended.
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:4rem; text-align:center; font-size:0.6rem; color:#4b5563;
                letter-spacing:0.12em; text-transform:uppercase;
                border-top:1px solid rgba(99,102,241,0.08); padding-top:1.2rem;'>
        &copy; 2026 Haider Haroon &nbsp;|&nbsp;
        <span style='color:#a5b4fc;'>Scikit-Learn</span> &middot;
        <span style='color:#22d3ee;'>Lasso Logistic Regression</span> &middot;
        <span style='color:#6366f1;'>Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ABOUT THE STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'About the Study':

    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.4rem;
                font-weight:700; color:#a5b4fc; letter-spacing:0.04em; margin-bottom:6px;'>
        About the Study
    </div>
    <div style='width:50px; height:2px;
                background:linear-gradient(90deg,transparent,#6366f1,transparent);
                margin-bottom:20px;'></div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#5a7a76; font-size:0.85rem; line-height:1.65; margin-bottom:1.8rem;'>"
        "A breakdown of the dataset, model pipeline, and why this kind of tool matters.</p>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='card'>
        <div class='section-label'>Why It Matters</div>
        <p style='color:#e2f0ee; font-size:0.9rem; line-height:1.85; margin:0;'>
            Breast cancer is one of the most common cancers worldwide.
            <strong style='color:#2dd4bf;'>Early detection</strong> is the single biggest factor
            in improving survival rates — and not everyone has access to specialist diagnostics.
            This project shows how a simple ML model trained on measurable biomarkers can assist
            in the classification process quickly and reliably.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='card'>
            <div class='section-label' style='color:#10b981;'>Dataset</div>
            <div class='stat-row'>
                <span class='stat-icon'>🗂️</span>
                <div><div class='stat-label'>Source</div>
                     <div class='stat-value'>Breast Cancer Wisconsin</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>🔢</span>
                <div><div class='stat-label'>Samples</div>
                     <div class='stat-value'>569 patient records</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>📐</span>
                <div><div class='stat-label'>Features</div>
                     <div class='stat-value'>30 numeric biomarkers</div></div>
            </div>
            <div class='stat-row' style='border:none;'>
                <span class='stat-icon'>🏷️</span>
                <div><div class='stat-label'>Classes</div>
                     <div class='stat-value'>Malignant &middot; Benign</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <div class='section-label' style='color:#f43f5e;'>Model Pipeline</div>
            <div class='stat-row'>
                <span class='stat-icon'>🔬</span>
                <div><div class='stat-label'>Algorithm</div>
                     <div class='stat-value'>Lasso Logistic Regression</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>⚖️</span>
                <div><div class='stat-label'>Preprocessing</div>
                     <div class='stat-value'>StandardScaler + Encoding</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>🧪</span>
                <div><div class='stat-label'>Validation</div>
                     <div class='stat-value'>5-Fold Cross Validation</div></div>
            </div>
            <div class='stat-row' style='border:none;'>
                <span class='stat-icon'>📦</span>
                <div><div class='stat-label'>Saved With</div>
                     <div class='stat-value'>joblib (.pkl)</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='padding:14px 18px; background:rgba(245,158,11,0.06);
                border:1px solid rgba(245,158,11,0.2); border-radius:10px;
                font-size:0.8rem; color:#c8a96a; line-height:1.7;'>
        &#9888;&#65039; <strong style='color:#f59e0b;'>Disclaimer:</strong>
        This tool is for educational and research purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ABOUT US
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'About Us':

    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.4rem;
                font-weight:800; color:#a5b4fc; letter-spacing:0.04em; margin-bottom:6px;'>
        About the Developer
    </div>
    <div style='width:50px; height:2px;
                background:linear-gradient(90deg,transparent,#6366f1,transparent);
                margin-bottom:28px;'></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='border-color:rgba(99,102,241,0.3);
         background:linear-gradient(135deg,rgba(99,102,241,0.07),rgba(34,211,238,0.04));'>
        <div style='display:flex; flex-direction:column; gap:10px;'>
            <div style='font-family:"Syne",sans-serif; font-size:1.55rem; font-weight:800;
                        background:linear-gradient(90deg,#a5b4fc,#22d3ee);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        letter-spacing:0.02em;'>
                Haider Haroon
            </div>
            <div style='font-size:0.72rem; color:#6366f1; letter-spacing:0.2em;
                        text-transform:uppercase; font-weight:600;'>
                AI Engineer &nbsp;·&nbsp; ML Developer
            </div>
            <div style='width:40px; height:1px;
                        background:linear-gradient(90deg,#6366f1,transparent);
                        margin:4px 0 8px;'></div>
            <p style='color:#e8eaf6; font-size:0.92rem; line-height:1.9; margin:0 0 10px;'>
                I'm <strong style='color:#a5b4fc;'>Haider Haroon</strong> — an AI engineer who turns
                data into decisions. I build end-to-end machine learning systems that don't just run
                in notebooks — they ship. From crafting clean pipelines to designing interfaces people
                actually want to use, I care about the full stack of the problem.
            </p>
            <p style='color:#4b5563; font-size:0.87rem; line-height:1.85; margin:0;'>
                This project — CancerSense — is one example of that: a real clinical dataset, a rigorously
                validated model, and a UI that makes it accessible. I believe the best ML work is the kind
                that reaches people beyond the terminal.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class='card'>
            <div class='section-label' style='color:#22d3ee;'>What I Build</div>
            <div style='display:flex; flex-direction:column; gap:14px;'>
                <div style='display:flex; gap:12px; align-items:flex-start;'>
                    <div style='width:6px; height:6px; border-radius:50%; background:#a5b4fc;
                                margin-top:7px; flex-shrink:0;'></div>
                    <div>
                        <div style='color:#e8eaf6; font-size:0.88rem; font-weight:600;'>Machine Learning Systems</div>
                        <div style='color:#4b5563; font-size:0.8rem; line-height:1.6; margin-top:3px;'>
                            Classification, regression, clustering — with clean preprocessing pipelines and proper validation.
                        </div>
                    </div>
                </div>
                <div style='display:flex; gap:12px; align-items:flex-start;'>
                    <div style='width:6px; height:6px; border-radius:50%; background:#22d3ee;
                                margin-top:7px; flex-shrink:0;'></div>
                    <div>
                        <div style='color:#e8eaf6; font-size:0.88rem; font-weight:600;'>Data-Driven Applications</div>
                        <div style='color:#4b5563; font-size:0.8rem; line-height:1.6; margin-top:3px;'>
                            Interactive tools built with Streamlit, Python, and real-world datasets that solve actual problems.
                        </div>
                    </div>
                </div>
                <div style='display:flex; gap:12px; align-items:flex-start;'>
                    <div style='width:6px; height:6px; border-radius:50%; background:#6366f1;
                                margin-top:7px; flex-shrink:0;'></div>
                    <div>
                        <div style='color:#e8eaf6; font-size:0.88rem; font-weight:600;'>End-to-End Pipelines</div>
                        <div style='color:#4b5563; font-size:0.8rem; line-height:1.6; margin-top:3px;'>
                            From raw data ingestion to model deployment — every step accounted for.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <div class='section-label' style='color:#a5b4fc;'>Background</div>
            <div class='stat-row'>
                <div>
                    <div class='stat-label'>Role</div>
                    <div class='stat-value' style='color:#a5b4fc;'>AI Engineer</div>
                </div>
            </div>
            <div class='stat-row'>
                <div>
                    <div class='stat-label'>Focus</div>
                    <div class='stat-value' style='color:#22d3ee;'>ML &middot; Data Science</div>
                </div>
            </div>
            <div class='stat-row' style='border:none;'>
                <div>
                    <div class='stat-label'>Based In</div>
                    <div class='stat-value' style='color:#a5b4fc;'>Pakistan</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='section-label' style='color:#22d3ee;'>Tech Stack Used in This Project</div>
        <div style='display:flex; flex-wrap:wrap; gap:10px;'>
            <span style='background:rgba(165,180,252,0.1); border:1px solid rgba(165,180,252,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#a5b4fc;'>Python</span>
            <span style='background:rgba(34,211,238,0.1); border:1px solid rgba(34,211,238,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#22d3ee;'>Scikit-Learn</span>
            <span style='background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#818cf8;'>Lasso Regression</span>
            <span style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#f59e0b;'>Pandas &middot; NumPy</span>
            <span style='background:rgba(165,180,252,0.1); border:1px solid rgba(165,180,252,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#a5b4fc;'>Streamlit</span>
            <span style='background:rgba(34,211,238,0.1); border:1px solid rgba(34,211,238,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#22d3ee;'>joblib</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='section-label' style='color:#f43f5e;'>Find Me Online</div>
        <div style='display:flex; gap:16px; flex-wrap:wrap; align-items:center;'>
            <a href='https://www.linkedin.com/in/haider-haroon-8a0209306/' target='_blank'
               style='display:inline-flex; align-items:center; gap:8px; text-decoration:none;
                      background:rgba(10,102,194,0.1); border:1px solid rgba(10,102,194,0.3);
                      border-radius:10px; padding:10px 18px;
                      color:#60a5fa; font-weight:600; font-size:0.88rem;
                      transition:all 0.2s;'>
                <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'
                     viewBox='0 0 24 24' fill='#60a5fa'>
                    <path d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/>
                </svg>
                LinkedIn
            </a>
            <a href='https://github.com/Haid3rH' target='_blank'
               style='display:inline-flex; align-items:center; gap:8px; text-decoration:none;
                      background:rgba(165,180,252,0.08); border:1px solid rgba(165,180,252,0.25);
                      border-radius:10px; padding:10px 18px;
                      color:#a5b4fc; font-weight:600; font-size:0.88rem;
                      transition:all 0.2s;'>
                <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'
                     viewBox='0 0 24 24' fill='#a5b4fc'>
                    <path d='M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12'/>
                </svg>
                GitHub
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
