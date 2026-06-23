"""Tests unitaires pour le chargement des données."""
import pytest
import pandas as pd
import os


def test_train_csv_exists():
    """Vérifie que le fichier train existe."""
    path = 'data/train_cleaned.csv'
    assert os.path.exists(path), f"Le fichier {path} doit exister"


def test_test_csv_exists():
    """Vérifie que le fichier test existe."""
    path = 'data/test_cleaned.csv'
    assert os.path.exists(path), f"Le fichier {path} doit exister"


def test_results_csv_exists():
    """Vérifie que le fichier de résultats existe."""
    path = 'dashboard/data/all_models_comparison.csv'
    assert os.path.exists(path), f"Le fichier {path} doit exister"


def test_train_has_required_columns():
    """Vérifie que le train set a les colonnes requises."""
    df = pd.read_csv('data/train_cleaned.csv')
    required = ['target', 'text_clean']
    for col in required:
        assert col in df.columns, f"La colonne {col} manque dans train"


def test_target_is_binary():
    """Vérifie que la cible est bien binaire (0 ou 1)."""
    df = pd.read_csv('data/train_cleaned.csv')
    unique_values = sorted(df['target'].unique())
    assert unique_values == [0, 1], "Target doit être binaire (0/1)"


def test_train_test_split_size():
    """Vérifie les tailles approximatives du split 80/20."""
    train = pd.read_csv('data/train_cleaned.csv')
    test = pd.read_csv('data/test_cleaned.csv')
    total = len(train) + len(test)
    train_ratio = len(train) / total
    assert 0.75 < train_ratio < 0.85, f"Ratio train inattendu: {train_ratio}"


def test_no_nan_in_target():
    """Vérifie qu'il n'y a pas de NaN dans la cible."""
    df = pd.read_csv('data/train_cleaned.csv')
    assert df['target'].isna().sum() == 0, "La cible ne doit pas avoir de NaN"


def test_models_comparison_has_30_models():
    """Vérifie que le tableau de comparaison contient 30 modèles."""
    df = pd.read_csv('dashboard/data/all_models_comparison.csv')
    assert len(df) == 30, f"Devrait avoir 30 modèles, trouvé {len(df)}"