"""Tests unitaires pour les calculs de métriques."""
import pytest
from sklearn.metrics import f1_score, recall_score, precision_score


def calculate_f2(precision, recall):
    """Calcule le F2-score à partir de precision et recall."""
    if precision + recall == 0:
        return 0
    return (5 * precision * recall) / (4 * precision + recall)


def test_f2_perfect_prediction():
    """F2 doit être 1.0 quand P=R=1."""
    assert calculate_f2(1.0, 1.0) == 1.0


def test_f2_zero_when_both_zero():
    """F2 doit être 0 quand P=R=0."""
    assert calculate_f2(0, 0) == 0


def test_f2_prioritizes_recall():
    """F2 doit favoriser le recall plus que le F1."""
    # Cas: precision haute, recall bas
    p_high_r_low = calculate_f2(0.9, 0.5)
    # Cas: precision basse, recall haut
    p_low_r_high = calculate_f2(0.5, 0.9)
    assert p_low_r_high > p_high_r_low, "F2 doit privilégier le recall"


def test_f2_roberta_score():
    """Vérifie le F2 de RoBERTa avec les valeurs connues."""
    # Recall=0.814, Precision=0.732
    f2 = calculate_f2(0.732, 0.814)
    assert abs(f2 - 0.796) < 0.01, f"F2 RoBERTa devrait être ~0.796, obtenu {f2}"


def test_f1_calculation():
    """Test simple du F1-score."""
    y_true = [0, 1, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 1, 0]
    f1 = f1_score(y_true, y_pred)
    assert 0 <= f1 <= 1, "F1 doit être entre 0 et 1"