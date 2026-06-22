import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Analyse des erreurs", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Analyse des erreurs du modèle RoBERTa")
st.markdown("---")

st.markdown("""
Cette section analyse en détail les erreurs commises par notre meilleur modèle
(**RoBERTa**, F1=0.771) sur le test set, afin de comprendre ses limites et
identifier des pistes d'amélioration.
""")

# ── MATRICE DE CONFUSION ──────────────────────────────────────────────────────
st.markdown("## 📊 Matrice de confusion")

@st.cache_data
def load_cm():
    return pd.read_csv('dashboard/data/confusion_matrix_roberta.csv', index_col=0)

cm = load_cm()

col1, col2 = st.columns([1, 1])

with col1:
    fig = go.Figure(data=go.Heatmap(
        z=cm.values,
        x=cm.columns.tolist(),
        y=cm.index.tolist(),
        text=cm.values,
        texttemplate='%{text}',
        textfont={"size": 24},
        colorscale='Blues',
        showscale=False
    ))
    fig.update_layout(
        title='Matrice de confusion — RoBERTa',
        height=400,
        xaxis_title='Prédiction',
        yaxis_title='Réalité'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Décodage des résultats")
    
    tn, fp = cm.iloc[0, 0], cm.iloc[0, 1]
    fn, tp = cm.iloc[1, 0], cm.iloc[1, 1]
    
    st.markdown(f"""
    | Type | Nombre | Description |
    |---|---|---|
    | ✅ **Vrais négatifs (TN)** | {tn} | Non-disasters correctement identifiés |
    | ✅ **Vrais positifs (TP)** | {tp} | Disasters correctement détectés |
    | ❌ **Faux positifs (FP)** | {fp} | Fausses alertes (non-disasters → disaster) |
    | ❌ **Faux négatifs (FN)** | {fn} | Disasters ratés (disaster → non-disaster) ⚠️ |
    """)
    
    total = tn + fp + fn + tp
    st.metric("Total de tweets testés", f"{total:,}")
    st.metric("Taux d'erreur global", f"{(fp+fn)/total*100:.2f}%")

st.markdown("---")

# ── INTERPRÉTATION ────────────────────────────────────────────────────────────
st.markdown("## 💡 Interprétation")

col1, col2 = st.columns(2)

with col1:
    st.error(f"""
    ### ⚠️ Faux négatifs ({fn} tweets)
    
    **Disasters ratés** — le modèle a classifié ces tweets comme non-disasters
    alors qu'ils décrivent de vraies catastrophes.
    
    **Conséquence opérationnelle :** ce sont les erreurs les plus graves pour
    un système d'alerte — chaque FN représente une vraie catastrophe non détectée
    qui n'aurait pas déclenché d'alerte.
    
    **Taux de FN** : {fn/(fn+tp)*100:.1f}% des disasters ratés
    """)

with col2:
    st.warning(f"""
    ### 🚨 Faux positifs ({fp} tweets)
    
    **Fausses alertes** — le modèle a classifié ces tweets comme disasters
    alors qu'ils ne décrivent pas de catastrophes réelles.
    
    **Conséquence opérationnelle :** moins grave que les FN — une vérification
    humaine peut filtrer ces fausses alertes avant action.
    
    **Taux de FP** : {fp/(fp+tn)*100:.1f}% des non-disasters mal classifiés
    """)

st.markdown("---")

# ── EXEMPLES D'ERREURS ────────────────────────────────────────────────────────
st.markdown("## 🔎 Exemples d'erreurs")

@st.cache_data
def load_errors():
    fn_df = pd.read_csv('dashboard/data/false_negatives_roberta.csv')
    fp_df = pd.read_csv('dashboard/data/false_positives_roberta.csv')
    return fn_df, fp_df

fn_df, fp_df = load_errors()

tab1, tab2 = st.tabs(["❌ Faux négatifs (Disasters ratés)", "🚨 Faux positifs (Fausses alertes)"])

with tab1:
    st.markdown("""
    **Top 20 des disasters ratés** — tweets décrivant de vraies catastrophes
    mais classés à tort comme non-disasters par le modèle.
    """)
    
    fn_display = fn_df.copy()
    fn_display['probability_disaster'] = fn_display['probability_disaster'].round(3)
    fn_display.columns = ['Texte', 'Keyword', 'Probabilité disaster']
    st.dataframe(fn_display, use_container_width=True, hide_index=True, height=500)
    
    st.info("""
    💡 **Pistes d'amélioration des FN :**
    - Tweets souvent **courts** ou **ambigus** linguistiquement
    - **Vocabulaire spécifique** mal capté par le modèle
    - Pourrait bénéficier de **data augmentation** pour la classe minoritaire
    - **Abaisser le seuil de décision** (ex: 0.4 au lieu de 0.5) augmenterait le recall
    """)

with tab2:
    st.markdown("""
    **Top 20 des fausses alertes** — tweets ne décrivant pas de vraies catastrophes
    mais classés à tort comme disasters par le modèle.
    """)
    
    fp_display = fp_df.copy()
    fp_display['probability_disaster'] = fp_display['probability_disaster'].round(3)
    fp_display.columns = ['Texte', 'Keyword', 'Probabilité disaster']
    st.dataframe(fp_display, use_container_width=True, hide_index=True, height=500)
    
    st.info("""
    💡 **Pistes d'amélioration des FP :**
    - Usages **métaphoriques** non détectés (ex: "this party is fire")
    - Contextes **humoristiques** ou **sarcastiques**
    - Mots-clés sensibles ("bomb", "fire") utilisés hors contexte catastrophe
    - Pourrait bénéficier d'un **modèle plus contextuel** ou de **règles métier**
    """)

st.markdown("---")

# ── CONCLUSION ────────────────────────────────────────────────────────────────
st.markdown("## 🎯 Conclusion de l'analyse")

st.success(f"""
### Performance globale satisfaisante mais perfectible

Avec **{fn} disasters ratés** sur 415 vrais disasters dans le test set, le modèle
RoBERTa atteint un **recall de {tp/(tp+fn)*100:.1f}%** — il détecte la grande
majorité des vraies catastrophes.

Les **{fp} fausses alertes** sur 1830 non-disasters représentent un **taux d'erreur
de {fp/(fp+tn)*100:.1f}%** — gérable avec une couche de vérification humaine
dans un déploiement réel.

**Recommandations pour un déploiement opérationnel :**
1. Ajuster le **seuil de décision** selon le contexte (recall vs precision)
2. Intégrer une **équipe de vérification humaine** pour les alertes critiques
3. Surveiller les **dérives de performance** sur de nouveaux types de tweets
""")