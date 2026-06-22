import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Comparaison", page_icon="📈", layout="wide")

st.markdown("# 📈 Comparaison approfondie")
st.markdown("---")

st.markdown("""
Cette section propose plusieurs visualisations pour comparer les modèles sous
différents angles : par notebook, par métrique, et par catégorie d'approche.
""")

@st.cache_data
def load_data():
    return pd.read_csv('data/all_models_comparison.csv')

all_results = load_data()

color_map = {
    'TF-IDF Classique' : '#3498DB',
    'TF-IDF Avancé'    : '#2ECC71',
    'Embeddings'       : '#F39C12',
    'Ensemblistes'     : '#E74C3C',
    'Deep Learning'    : '#9B59B6'
}

# ── 1. MEILLEUR PAR NOTEBOOK ─────────────────────────────────────────────────
st.markdown("## 🏆 Meilleur modèle de chaque notebook")

best_per_nb = all_results.loc[all_results.groupby('notebook')['test_f1_class_1'].idxmax()]
best_per_nb = best_per_nb.sort_values('test_f1_class_1', ascending=True)

fig = px.bar(
    best_per_nb,
    x='test_f1_class_1',
    y='notebook',
    orientation='h',
    color='categorie',
    text='pipeline',
    color_discrete_map=color_map,
    labels={'test_f1_class_1': 'F1 classe 1', 'notebook': ''}
)
fig.update_traces(textposition='outside')
fig.update_layout(height=400, xaxis_range=[0.55, 0.85])
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
💡 Le **Deep Learning** domine clairement avec RoBERTa (F1=0.771).
Les autres notebooks plafonnent autour de **F1 ≈ 0.71**, marquant la limite
des approches non-Transformers sur ce dataset.
""")

st.markdown("---")

# ── 2. BOXPLOT PAR CATÉGORIE ──────────────────────────────────────────────────
st.markdown("## 📊 Distribution du F1 par catégorie")

fig = px.box(
    all_results,
    x='categorie',
    y='test_f1_class_1',
    color='categorie',
    points='all',
    hover_data=['pipeline'],
    color_discrete_map=color_map,
    labels={'test_f1_class_1': 'F1 classe 1', 'categorie': 'Catégorie'}
)
fig.update_layout(height=500, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
💡 **Lecture du boxplot :**
- La **boîte** représente 50% des modèles (Q1 à Q3)
- La **ligne médiane** dans la boîte = médiane
- Les **points** = modèles individuels
- Plus la boîte est petite, plus la catégorie est consistante
""")

st.markdown("---")

# ── 3. HEATMAP DES MÉTRIQUES ──────────────────────────────────────────────────
st.markdown("## 🔥 Heatmap multi-métriques (TOP 10)")

top10 = all_results.head(10)
heatmap_data = top10[['pipeline', 'test_f1_class_1', 'test_recall_class_1',
                       'test_precision_class_1', 'test_accuracy',
                       'test_balanced_accuracy']].set_index('pipeline')

heatmap_data.columns = ['F1', 'Recall', 'Precision', 'Accuracy', 'Balanced Acc.']

fig = px.imshow(
    heatmap_data,
    color_continuous_scale='RdYlGn',
    aspect='auto',
    text_auto='.3f',
    range_color=[0.5, 0.95],
    labels=dict(x='Métrique', y='Modèle', color='Score')
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
💡 **Lecture de la heatmap :**
- **Rouge** = score faible (< 0.6)
- **Jaune** = score moyen (~ 0.7)
- **Vert** = score élevé (> 0.85)

Les **transformers** (RoBERTa, BERTweet, DistilBERT) dominent toutes les métriques.
""")

st.markdown("---")

# ── 4. PRECISION VS RECALL ────────────────────────────────────────────────────
st.markdown("## ⚖️ Trade-off Precision / Recall")

fig = px.scatter(
    all_results,
    x='test_recall_class_1',
    y='test_precision_class_1',
    size='test_f1_class_1',
    color='categorie',
    hover_data=['pipeline', 'test_f1_class_1'],
    color_discrete_map=color_map,
    labels={'test_recall_class_1': 'Recall', 'test_precision_class_1': 'Precision'}
)

# Lignes de référence
fig.add_shape(type='line', x0=0.5, y0=0.5, x1=0.85, y1=0.85,
              line=dict(color='gray', dash='dash'))
fig.update_layout(height=550)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
💡 **Interprétation :**
- **Diagonale** = modèles équilibrés (precision ≈ recall)
- **Au-dessus de la diagonale** = modèles privilégiant la precision
- **En-dessous de la diagonale** = modèles privilégiant le recall
- Pour un système d'alerte, on préfère un **recall élevé** (ne pas manquer de disasters)
""")

st.markdown("---")

# ── 5. RADAR CHART TOP 3 ──────────────────────────────────────────────────────
st.markdown("## 🎯 Comparaison radar du TOP 3")

top3 = all_results.head(3)
metrics = ['test_f1_class_1', 'test_recall_class_1', 'test_precision_class_1',
           'test_accuracy', 'test_balanced_accuracy']
labels = ['F1', 'Recall', 'Precision', 'Accuracy', 'Balanced Acc.']

fig = go.Figure()
colors_radar = ['#9B59B6', '#3498DB', '#E74C3C']

for idx, (_, row) in enumerate(top3.iterrows()):
    values = [row[m] for m in metrics]
    values += values[:1]  # fermer le radar
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels + [labels[0]],
        fill='toself',
        name=row['pipeline'],
        line=dict(color=colors_radar[idx], width=2)
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0.6, 1.0])),
    showlegend=True,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
💡 **Le radar chart** permet de visualiser les forces et faiblesses de chaque modèle.
Plus la forme est grande et équilibrée, meilleur est le modèle global.
""")

st.markdown("---")

# ── 6. ÉVOLUTION DES APPROCHES ────────────────────────────────────────────────
st.markdown("## 📅 Évolution des approches NLP")

evolution_data = pd.DataFrame({
    'Année'  : ['2014', '2015', '2017', '2018', '2019'],
    'Modèle' : ['TextCNN', 'BiLSTM', 'TF-IDF + LogReg', 'BERT/BERTweet/DistilBERT', 'RoBERTa'],
    'F1 max' : [0.615, 0.616, 0.712, 0.747, 0.771],
    'Type'   : ['CNN', 'RNN', 'TF-IDF', 'Transformer', 'Transformer']
})

fig = px.line(
    evolution_data,
    x='Année',
    y='F1 max',
    markers=True,
    text='Modèle',
    hover_data=['Type']
)
fig.update_traces(textposition='top center', marker=dict(size=15))
fig.update_layout(height=500, yaxis_range=[0.55, 0.85])
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
💡 **L'évolution du NLP est clairement visible :**
- **Pré-2018** : CNN et RNN plafonnent à ~0.62
- **2018-2019** : révolution des Transformers (BERT)
- **2019+** : améliorations continues (RoBERTa, BERTweet)

Le **gap de +15 points de F1** entre les Transformers et les approches précédentes
illustre la rupture méthodologique introduite par l'attention bidirectionnelle.
""")