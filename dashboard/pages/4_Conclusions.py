import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Conclusions", page_icon="🏆", layout="wide")

st.markdown("# 🏆 Conclusions & Sélection du modèle final")
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv('data/all_models_comparison.csv')

all_results = load_data()
best_model = all_results.iloc[0]

# ── MODÈLE GAGNANT ────────────────────────────────────────────────────────────
st.markdown("## 🥇 Modèle final sélectionné")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 1rem; color: white; text-align: center;'>
        <h1 style='margin: 0; font-size: 3rem;'>{best_model['pipeline']}</h1>
        <p style='font-size: 1.2rem; margin-top: 0.5rem;'>Meilleur modèle du projet</p>
        <hr style='border-color: rgba(255,255,255,0.3);'>
        <h2 style='margin: 0;'>F1 = {best_model['test_f1_class_1']:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Métriques détaillées")

    metrics_data = {
        'Métrique'          : ['F1 classe 1', 'Recall', 'Precision', 'Accuracy', 'Balanced Accuracy'],
        'Score'             : [
            f"{best_model['test_f1_class_1']:.3f}",
            f"{best_model['test_recall_class_1']:.3f}",
            f"{best_model['test_precision_class_1']:.3f}",
            f"{best_model['test_accuracy']:.3f}",
            f"{best_model['test_balanced_accuracy']:.3f}"
        ],
        'Interprétation'    : [
            'Équilibre précision/rappel',
            '81% des disasters sont détectés',
            '73% des prédictions disaster sont correctes',
            '91% des prédictions sont correctes',
            'Performance équilibrée entre les deux classes'
        ]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

st.markdown("---")

# ── POURQUOI ROBERTA ──────────────────────────────────────────────────────────
st.markdown("## 🤔 Pourquoi RoBERTa ?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Forces de RoBERTa

    **1. Pré-entraînement massif**
    RoBERTa a été pré-entraîné sur **160 GB de texte** (vs 16 GB pour BERT),
    ce qui lui donne une compréhension plus riche du langage naturel.

    **2. Attention bidirectionnelle**
    Contrairement à BiLSTM qui lit séquentiellement, RoBERTa utilise
    l'**attention pour considérer tous les mots simultanément** —
    capture le contexte global d'un coup.

    **3. Sous-mots (BPE)**
    Le tokeniseur BPE découpe les mots inconnus en sous-mots connus,
    gérant mieux le vocabulaire Twitter (hashtags, abréviations).

    **4. Optimisations vs BERT**
    - Suppression de Next Sentence Prediction
    - Batches plus grands pendant le pré-entraînement
    - Plus de données et plus d'epochs
    """)

with col2:
    st.markdown("""
    ### Pourquoi pas BERTweet ?

    Surprenant — **BERTweet** est pourtant spécialisé tweets, mais RoBERTa
    le dépasse de **+0.024 points de F1**.

    **Explications possibles :**
    - BERTweet est plus petit (135M vs RoBERTa 125M)
    - Le pré-entraînement plus poussé de RoBERTa compense la spécialisation
    - Notre dataset (~9 000 tweets) bénéficie plus d'un modèle général robuste

    ### Pourquoi pas BiLSTM ou TextCNN ?

    Ces architectures **pré-Transformers** plafonnent à F1 ≈ 0.615 :
    - **BiLSTM** : lecture séquentielle, oublie le début sur longues séquences
    - **TextCNN** : capture seulement des n-grams locaux, pas le contexte global

    Le **gap de +15 points de F1** illustre la révolution des Transformers.
    """)

st.markdown("---")

# ── COMPARAISON TOP 3 ─────────────────────────────────────────────────────────
st.markdown("## 🥉 Comparaison du TOP 3")

top3 = all_results.head(3)

fig = go.Figure()
metrics = ['test_f1_class_1', 'test_recall_class_1', 'test_precision_class_1',
           'test_accuracy', 'test_balanced_accuracy']
labels = ['F1', 'Recall', 'Precision', 'Accuracy', 'Balanced Acc.']

for idx, (_, row) in enumerate(top3.iterrows()):
    fig.add_trace(go.Bar(
        name=row['pipeline'],
        x=labels,
        y=[row[m] for m in metrics],
        text=[f"{row[m]:.3f}" for m in metrics],
        textposition='outside'
    ))

fig.update_layout(
    barmode='group',
    height=500,
    yaxis_range=[0.5, 1.0],
    yaxis_title='Score',
    legend_title='Modèle'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── PRINCIPAUX APPRENTISSAGES ─────────────────────────────────────────────────
st.markdown("## 💡 Principaux apprentissages")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎯 Sur la tâche

    - **F1 plafonné à ~0.71** avec les approches classiques
    - **Transformers dépassent 0.74** systématiquement
    - **Recall = priorité** pour un système d'alerte
    - Le **déséquilibre 81/19** impose une métrique adaptée
    """)

with col2:
    st.markdown("""
    ### 🛠️ Sur les approches

    - **TF-IDF + LogReg** : excellente baseline
    - **Embeddings moyennés** : perdent le contexte
    - **Ensemblistes** : compromis intéressant
    - **Transformers** : état de l'art incontesté
    """)

with col3:
    st.markdown("""
    ### 🚀 Sur le projet

    - **30 modèles** comparés rigoureusement
    - **GridSearchCV + K-Fold** systématique
    - **Pipeline reproductible** sklearn
    - **Déploiement Streamlit + Hugging Face**
    """)

st.markdown("---")

# ── DÉPLOIEMENT ───────────────────────────────────────────────────────────────
st.markdown("## 🚀 Déploiement")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Dashboard Streamlit (actuel)

    Cette interface présente l'**analyse comparative complète** :
    - Vue d'ensemble du projet
    - Analyse exploratoire et preprocessing
    - Tableau interactif de tous les modèles
    - Visualisations comparatives
    - Conclusions et sélection finale

    Hébergé sur **Streamlit Community Cloud** (gratuit).
    """)

with col2:
    st.markdown("""
    ### Interface de prédiction (Hugging Face Spaces)

    Une **interface interactive** permet de tester le modèle RoBERTa
    sur de nouveaux tweets :
    1. Saisir un tweet
    2. Obtenir une prédiction (disaster / non-disaster)
    3. Voir les probabilités et l'explication

    🔗 **URL** : `https://huggingface.co/spaces/...` (à déployer)
    """)

st.markdown("---")

# ── PISTES D'AMÉLIORATION ─────────────────────────────────────────────────────
st.markdown("## 🔮 Pistes d'amélioration")

st.markdown("""
Plusieurs directions pourraient encore améliorer les performances :

1. **Data augmentation** : back-translation, synonymes, paraphrases pour enrichir
   la classe minoritaire (disaster)

2. **Modèles plus grands** : RoBERTa-large, DeBERTa-v3-large (avec GPU plus puissant)

3. **Fine-tuning prolongé** : plus d'epochs avec learning rate scheduling avancé

4. **Ensemble de Transformers** : combiner RoBERTa + BERTweet + DeBERTa par voting

5. **Domain adaptation** : pré-entraîner sur des tweets d'actualités/urgences spécifiquement

6. **Multimodal** : intégrer les images souvent associées aux tweets de catastrophes
""")

st.markdown("---")

# ── CONCLUSION ────────────────────────────────────────────────────────────────
st.markdown("## 🎉 Conclusion")

st.success(f"""
### 🏆 RoBERTa est le modèle final retenu (F1 = {best_model['test_f1_class_1']:.3f})

Ce projet a permis d'explorer **toute la chaîne du NLP** —
des approches classiques TF-IDF aux Transformers modernes — sur un cas concret
de classification de tweets de catastrophe.

L'évolution des scores **0.625 → 0.771 (+0.146 points de F1)** illustre l'apport
décisif du Deep Learning et des Transformers en NLP, tout en confirmant que
les baselines classiques restent des références solides à dépasser.
""")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999;'>"
    "Projet réalisé dans le cadre du cours de Machine Learning 2 — ISE2 2026"
    "</p>",
    unsafe_allow_html=True
)