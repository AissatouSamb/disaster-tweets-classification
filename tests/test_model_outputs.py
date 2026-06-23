"""Tests sur les outputs et résultats des modèles."""
import pytest
import pandas as pd


def test_top_model_is_roberta():
    """Vérifie que le meilleur modèle est RoBERTa."""
    df = pd.read_csv('dashboard/data/all_models_comparison.csv')
    df_sorted = df.sort_values('test_f1_class_1', ascending=False)
    best = df_sorted.iloc[0]
    assert best['pipeline'] == 'RoBERTa', f"Meilleur modèle: {best['pipeline']}"


def test_top_f1_above_threshold():
    """Vérifie que le meilleur F1 dépasse 0.75."""
    df = pd.read_csv('dashboard/data/all_models_comparison.csv')
    assert df['test_f1_class_1'].max() > 0.75, "Le meilleur F1 doit être > 0.75"


def test_all_metrics_in_valid_range():
    """Vérifie que toutes les métriques sont entre 0 et 1."""
    df = pd.read_csv('dashboard/data/all_models_comparison.csv')
    metric_cols = ['test_f1_class_1', 'test_recall_class_1',
                   'test_precision_class_1', 'test_accuracy']
    for col in metric_cols:
        assert (df[col] >= 0).all(), f"{col} contient des valeurs négatives"
        assert (df[col] <= 1).all(), f"{col} contient des valeurs > 1"


def test_categories_complete():
    """Vérifie que toutes les catégories d'approches sont représentées."""
    df = pd.read_csv('dashboard/data/all_models_comparison.csv')
    expected_categories = ['TF-IDF Classique', 'TF-IDF Avancé', 'Embeddings',
                            'Ensemblistes', 'Deep Learning']
    found = df['categorie'].unique()
    for cat in expected_categories:
        assert cat in found, f"Catégorie manquante: {cat}"


def test_f2_column_exists():
    """Vérifie que la colonne F2 a bien été ajoutée."""
    df = pd.read_csv('dashboard/data/all_models_comparison.csv')
    assert 'test_f2_class_1' in df.columns, "La colonne F2 doit exister"


def test_deep_learning_dominates():
    """Vérifie que le TOP 3 contient bien du Deep Learning."""
    df = pd.read_csv('dashboard/data/all_models_comparison.csv')
    top3 = df.sort_values('test_f1_class_1', ascending=False).head(3)
    dl_count = (top3['categorie'] == 'Deep Learning').sum()
    assert dl_count >= 2, "Au moins 2 modèles DL dans le TOP 3"