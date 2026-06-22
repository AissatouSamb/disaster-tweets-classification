import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Modèles", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Tous les modèles testés")
st.markdown("---")

st.markdown("""
Cette section présente l'**ensemble des 30 modèles testés** dans le projet,
organisés par notebook et catégorie d'approche.
""")

@st.cache_data
def load_data():
    return pd.read_csv('dashboard/data/all_models_comparison.csv')

all_results = load_data()

# ── SIDEBAR FILTRES ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Filtres")

categories = sorted(all_results['categorie'].unique())
selected_cat = st.sidebar.multiselect("Catégorie", options=categories, default=categories)

notebooks = sorted(all_results['notebook'].unique())
selected_nb = st.sidebar.multiselect("Notebook", options=notebooks, default=notebooks)

f1_min = st.sidebar.slider("F1 minimum", 0.0, 1.0, 0.0, 0.05)

filtered = all_results[
    (all_results['categorie'].isin(selected_cat)) &
    (all_results['notebook'].isin(selected_nb)) &
    (all_results['test_f1_class_1'] >= f1_min)
]

# ── MÉTRIQUES ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Modèles affichés", f"{len(filtered)} / {len(all_results)}")
with col2:
    if len(filtered) > 0:
        st.metric("Meilleur F1", f"{filtered['test_f1_class_1'].max():.3f}")
with col3:
    if len(filtered) > 0:
        st.metric("F1 moyen", f"{filtered['test_f1_class_1'].mean():.3f}")
with col4:
    if len(filtered) > 0:
        st.metric("F1 médian", f"{filtered['test_f1_class_1'].median():.3f}")

st.markdown("---")

# ── SCATTER PLOT ──────────────────────────────────────────────────────────────
st.markdown("## 📊 Visualisation interactive")

if len(filtered) > 0:
    color_map = {
        'TF-IDF Classique' : '#3498DB',
        'TF-IDF Avancé'    : '#2ECC71',
        'Embeddings'       : '#F39C12',
        'Ensemblistes'     : '#E74C3C',
        'Deep Learning'    : '#9B59B6'
    }
    
    fig = px.scatter(
        filtered,
        x='test_recall_class_1',
        y='test_precision_class_1',
        size='test_f1_class_1',
        color='categorie',
        hover_data=['pipeline', 'test_f1_class_1'],
        labels={'test_recall_class_1': 'Recall', 'test_precision_class_1': 'Precision'},
        color_discrete_map=color_map,
        title='Precision vs Recall (taille = F1)'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    💡 **Interprétation :**
    - **Haut-droite** : modèles équilibrés
    - **Haut-gauche** : modèles conservateurs (peu de faux positifs)
    - **Bas-droite** : modèles agressifs
    - **Taille des points** : F1-score
    """)

st.markdown("---")

# ── TABLEAU ───────────────────────────────────────────────────────────────────
st.markdown("## 📋 Tableau détaillé")

if len(filtered) > 0:
    display_df = filtered[['rang', 'pipeline', 'notebook', 'categorie',
                           'test_f1_class_1', 'test_f2_class_1',
                           'test_recall_class_1', 'test_precision_class_1',
                           'test_accuracy', 'test_balanced_accuracy']].copy()
    
    display_df.columns = ['Rang', 'Modèle', 'Notebook', 'Catégorie',
                          'F1', 'F2', 'Recall', 'Precision', 'Accuracy', 'Balanced Acc.']
    
    st.dataframe(
        display_df.style.background_gradient(
            subset=['F1', 'F2', 'Recall', 'Precision', 'Accuracy', 'Balanced Acc.'],
            cmap='RdYlGn',
            vmin=0.5, vmax=0.85
        ),
        use_container_width=True,
        hide_index=True,
        height=600
    )
else:
    st.warning("Aucun modèle ne correspond aux filtres.")

st.markdown("---")

# ── STATS PAR CATÉGORIE ───────────────────────────────────────────────────────
st.markdown("## 📈 Performance moyenne par catégorie")

cat_stats = all_results.groupby('categorie').agg({
    'test_f1_class_1': ['mean', 'max', 'min'],
    'test_f2_class_1': 'mean',
    'test_recall_class_1': 'mean',
    'test_precision_class_1': 'mean'
}).round(3)

cat_stats.columns = ['F1 moyen', 'F1 max', 'F1 min', 'F2 moyen', 'Recall moyen', 'Precision moyenne']
cat_stats = cat_stats.sort_values('F1 moyen', ascending=False)
st.dataframe(cat_stats, use_container_width=True)

st.markdown("---")
st.info("""
💡 **Observations clés :**
- Les **Deep Learning Transformers** dominent (F1 > 0.74)
- Les **TF-IDF classiques** restent compétitifs (F1 ≈ 0.71)
- Les **embeddings classiques** sont les moins performants
- Les **ensemblistes** offrent un bon compromis
""")