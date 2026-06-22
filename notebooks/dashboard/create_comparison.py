import pandas as pd

# Chargement des résultats de chaque notebook
nb1 = pd.read_csv('data/NB1_tuned_results.csv')
nb2 = pd.read_csv('data/NB2_tuned_results.csv')
nb3 = pd.read_csv('data/NB3_tuned_results.csv')
nb4 = pd.read_csv('data/NB4_tuned_results.csv')
nb5 = pd.read_csv('data/NB5_tuned_results.csv')
nb6 = pd.read_csv('data/NB6_all_dl_results.csv')

# Ajout des colonnes notebook et catégorie
nb1['notebook'] = 'NB1 — Count + TF-IDF Word'
nb2['notebook'] = 'NB2 — Char + Hybrid'
nb3['notebook'] = 'NB3 — Naive Bayes complet'
nb4['notebook'] = 'NB4 — Embeddings'
nb5['notebook'] = 'NB5 — Ensemblistes'
nb6['notebook'] = 'NB6 — Deep Learning'

nb1['categorie'] = 'TF-IDF Classique'
nb2['categorie'] = 'TF-IDF Avancé'
nb3['categorie'] = 'TF-IDF Classique'
nb4['categorie'] = 'Embeddings'
nb5['categorie'] = 'Ensemblistes'
nb6['categorie'] = 'Deep Learning'

# Colonnes communes
common_cols = [
    'pipeline', 'notebook', 'categorie',
    'test_f1_class_1', 'test_recall_class_1', 'test_precision_class_1',
    'test_accuracy', 'test_balanced_accuracy'
]

# Concaténation et tri
all_results = pd.concat([
    nb1[common_cols], nb2[common_cols], nb3[common_cols],
    nb4[common_cols], nb5[common_cols], nb6[common_cols]
], ignore_index=True)

all_results = all_results.sort_values('test_f1_class_1', ascending=False).reset_index(drop=True)
all_results['rang'] = all_results.index + 1
all_results = all_results.round(3)

# Sauvegarde
all_results.to_csv('data/all_models_comparison.csv', index=False)
print(f"✅ all_models_comparison.csv créé : {len(all_results)} modèles")