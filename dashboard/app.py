import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── CONFIGURATION DE LA PAGE ──────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Disaster Tweets Classifier",
    page_icon   = "🚨",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── STYLE CSS PERSONNALISÉ ────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🚨 Disaster Tweets Classifier</p>',
            unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center; color: #666;'>"
    "Détection automatique de tweets décrivant des catastrophes réelles"
    "</h4>",
    unsafe_allow_html=True
)
st.markdown("---")

# ── CHARGEMENT DES DONNÉES ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('dashboard/data/all_models_comparison.csv')

all_results = load_data()

# ── MÉTRIQUES CLÉS ────────────────────────────────────────────────────────────
st.markdown("## 📊 Aperçu du projet")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{len(all_results)}</div>
            <div class="stat-label">Modèles testés</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    best_f1 = all_results.iloc[0]['test_f1_class_1']
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{best_f1:.3f}</div>
            <div class="stat-label">Meilleur F1-score</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    best_model = all_results.iloc[0]['pipeline']
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number" style="font-size: 1.8rem;">{best_model}</div>
            <div class="stat-label">Meilleur modèle</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    n_categories = all_results['categorie'].nunique()
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{n_categories}</div>
            <div class="stat-label">Approches testées</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── DESCRIPTION DU PROJET ─────────────────────────────────────────────────────
st.markdown("## 🎯 Objectif du projet")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    En cas de catastrophe naturelle ou d'urgence, les médias sociaux comme Twitter
    deviennent une source précieuse d'information en temps réel. Cependant, tous les
    tweets contenant des mots-clés comme "fire", "earthquake" ou "flood" ne décrivent
    pas nécessairement de vraies catastrophes — beaucoup sont métaphoriques ou non-pertinents.

    **Ce projet vise à construire un système de classification automatique** pour distinguer :
    - 🚨 **Tweets décrivant de vraies catastrophes** (target = 1)
    - 💬 **Tweets non liés à des catastrophes** (target = 0)

    ### Approche

    Nous avons exploré **6 catégories d'approches NLP**, du plus simple au plus avancé :
    1. **TF-IDF Classique** — Vectorisation par fréquence pondérée
    2. **TF-IDF Avancé** — Caractères, n-grams hybrides
    3. **Embeddings** — Word2Vec, GloVe Twitter
    4. **Ensemblistes** — Random Forest, XGBoost, Voting, Stacking
    5. **Deep Learning** — BERT, BERTweet, RoBERTa, DistilBERT, BiLSTM, CNN
    """)

with col2:
    st.markdown("### Dataset")
    st.markdown("""
    - **Source** : Kaggle Disaster Tweets
    - **Total** : 11 224 tweets nettoyés
    - **Train** : 8 979 tweets (80%)
    - **Test** : 2 245 tweets (20%)
    - **Classes** :
        - Non-disaster : 81.4%
        - Disaster : 18.6%
    """)

    st.info("📌 Dataset déséquilibré — métrique principale : **F1 classe 1**")

st.markdown("---")

# ── TOP 5 MODÈLES (APERÇU) ────────────────────────────────────────────────────
st.markdown("## 🏆 TOP 5 des meilleurs modèles")

top5 = all_results.head(5)

fig = px.bar(
    top5,
    x        = 'test_f1_class_1',
    y        = 'pipeline',
    orientation = 'h',
    color    = 'categorie',
    text     = 'test_f1_class_1',
    color_discrete_map = {
        'TF-IDF Classique' : '#3498DB',
        'TF-IDF Avancé'    : '#2ECC71',
        'Embeddings'       : '#F39C12',
        'Ensemblistes'     : '#E74C3C',
        'Deep Learning'    : '#9B59B6'
    },
    labels   = {'test_f1_class_1': 'F1-score classe 1', 'pipeline': ''}
)
fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.update_layout(
    showlegend = True,
    height     = 400,
    xaxis_range= [0.65, 0.82],
    yaxis      = {'categoryorder': 'total ascending'}
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── NAVIGATION ────────────────────────────────────────────────────────────────
st.markdown("## 🧭 Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 📊 EDA
    Analyse exploratoire des données et preprocessing
    """)

with col2:
    st.markdown("""
    ### 🤖 Modèles
    Tableau interactif de tous les modèles testés
    """)

with col3:
    st.markdown("""
    ### 📈 Comparaison
    Visualisations comparatives par approche
    """)

with col4:
    st.markdown("""
    ### 🏆 Conclusions
    Analyse finale et sélection du meilleur modèle
    """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999;'>"
    "Projet réalisé dans le cadre du cours de Machine Learning 2 — ISE2 2026"
    "</p>",
    unsafe_allow_html=True
)