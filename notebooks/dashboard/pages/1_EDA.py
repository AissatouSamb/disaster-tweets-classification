import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Analyse Exploratoire & Preprocessing")
st.markdown("---")

# ── INTRODUCTION ──────────────────────────────────────────────────────────────
st.markdown("""
Cette section présente l'**analyse exploratoire (EDA)** du dataset Disaster Tweets
ainsi que les étapes de **preprocessing** appliquées pour préparer les données
à la modélisation.
""")

# ── DATASET OVERVIEW ──────────────────────────────────────────────────────────
st.markdown("## 📁 Vue d'ensemble du dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total tweets", "11 224", help="Après suppression des doublons")
with col2:
    st.metric("Train set", "8 979", "80%")
with col3:
    st.metric("Test set", "2 245", "20%")

st.markdown("""
Le dataset **Disaster Tweets** provient de Kaggle et a été collecté le 14 janvier 2020.
Il contient des tweets en anglais étiquetés selon qu'ils décrivent une vraie catastrophe
naturelle/urgence (target=1) ou non (target=0).
""")

# ── DISTRIBUTION DES CLASSES ──────────────────────────────────────────────────
st.markdown("## ⚖️ Distribution des classes")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    ### Déséquilibre modéré
    Le dataset présente un **déséquilibre 81/19** :
    - **Non-disaster** : 81.4%
    - **Disaster** : 18.6%

    Cette répartition implique qu'un modèle naïf qui prédirait toujours
    "non-disaster" obtiendrait 81% d'accuracy sans rien apprendre.

    **Métrique principale retenue : F1 classe 1** — elle force le modèle
    à équilibrer précision et rappel sur la classe disaster.
    """)

with col2:
    fig = go.Figure(data=[go.Pie(
        labels = ['Non-disaster', 'Disaster'],
        values = [81.4, 18.6],
        hole   = 0.5,
        marker = dict(colors=['#3498DB', '#E74C3C']),
        textinfo = 'label+percent',
        textfont = dict(size=16)
    )])
    fig.update_layout(
        height = 350,
        showlegend = False,
        annotations = [dict(text='Classes', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig, use_container_width=True)

# ── VARIABLES DU DATASET ──────────────────────────────────────────────────────
st.markdown("## 🔢 Variables du dataset")

variables_data = {
    'Variable'    : ['id', 'keyword', 'location', 'text', 'target'],
    'Type'        : ['int', 'string', 'string', 'string', 'binary'],
    'Description' : [
        'Identifiant unique du tweet',
        'Mot-clé associé au tweet (219 valeurs uniques)',
        "Localisation de l'auteur (30% NaN, abandonnée)",
        'Texte brut du tweet (variable principale)',
        'Cible binaire : 1 = disaster, 0 = non-disaster'
    ],
    'Utilisation' : ['Non utilisé', 'Target Encoding', 'Abandonnée', 'Variable principale', 'Cible']
}
st.dataframe(pd.DataFrame(variables_data), use_container_width=True, hide_index=True)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
st.markdown("## ⚙️ Feature Engineering")

st.markdown("""
À partir du texte brut, **8 features numériques** ont été créées pour capturer
les caractéristiques structurelles des tweets indépendamment de leur contenu :
""")

features_data = {
    'Feature'     : [
        'word_count', 'mean_word_length', 'lexical_diversity',
        'stop_word_count', 'url_count', 'punctuation_count',
        'hashtag_count', 'keyword_disaster_rate'
    ],
    'Description' : [
        'Nombre de mots dans le tweet',
        'Longueur moyenne des mots',
        'Ratio mots uniques / total (diversité lexicale)',
        'Nombre de stopwords (the, a, in,...)',
        "Présence d'URL (binaire 0/1)",
        'Nombre de signes de ponctuation',
        'Nombre de hashtags (#)',
        'Taux de disaster du mot-clé (Target Encoding)'
    ]
}
st.dataframe(pd.DataFrame(features_data), use_container_width=True, hide_index=True)

# ── PIPELINE DE NETTOYAGE ─────────────────────────────────────────────────────
st.markdown("## 🧹 Pipeline de nettoyage")

st.markdown("""
Le texte brut a été nettoyé en plusieurs étapes pour produire deux versions :
- **text_clean** : texte nettoyé (utilisé par les transformers comme BERT)
- **text_lemma** : texte nettoyé + lemmatisé (utilisé par les modèles classiques)
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Étapes de nettoyage
    1. Mise en minuscules
    2. Décodage des entités HTML
    3. Suppression des caractères corrompus
    4. Expansion des contractions (don't → do not)
    5. Correction des abréviations (lmao, MH370,...)
    6. Suppression des URLs
    7. Suppression des mentions @
    8. Suppression du symbole # (mot conservé)
    9. Suppression de la ponctuation
    10. Suppression des espaces multiples
    """)

with col2:
    st.markdown("""
    ### Lemmatisation (text_lemma)
    Appliquée via **spaCy** (en_core_web_sm) :
    - Suppression des stopwords
    - Tokens < 3 caractères supprimés
    - Lemmatisation (fires → fire)
    """)

    st.markdown("### Exemple")
    st.text("Avant : OMG! The fire was spreading FAST wildfire")
    st.text("Après : fire spread fast wildfire")

# ── DOUBLONS ──────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Gestion des doublons")

st.info("""
**146 doublons** ont été identifiés et supprimés du dataset original (11 370 → 11 224 tweets).
Ces doublons résultent de retweets ou de tweets très similaires partageant
le même contenu textuel.
""")

# ── CONCLUSION ────────────────────────────────────────────────────────────────
st.markdown("## ✅ Output du preprocessing")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Fichiers produits
    - train_cleaned.csv (8 979 × 17)
    - test_cleaned.csv (2 245 × 17)
    """)

with col2:
    st.markdown("""
    ### Colonnes finales (17)
    id, keyword, location, text, target,
    8 features numériques,
    text_clean, text_lemma, keyword_clean, text_tokens
    """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999;'>"
    "Page EDA — Disaster Tweets Classifier"
    "</p>",
    unsafe_allow_html=True
)