"""Tests unitaires pour le preprocessing des tweets."""
import pytest
import pandas as pd
import re


def clean_tweet(text):
    """Fonction simple de nettoyage de tweet."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def test_clean_tweet_removes_url():
    """Vérifie que les URLs sont supprimées."""
    result = clean_tweet("Check this https://example.com link")
    assert "https" not in result
    assert "example.com" not in result


def test_clean_tweet_removes_mentions():
    """Vérifie que les mentions @ sont supprimées."""
    result = clean_tweet("Hello @user how are you")
    assert "@user" not in result
    assert "hello" in result


def test_clean_tweet_removes_hashtag_symbol():
    """Vérifie que le # est supprimé mais le mot conservé."""
    result = clean_tweet("This is #fire")
    assert "#" not in result
    assert "fire" in result


def test_clean_tweet_lowercase():
    """Vérifie que le texte est en minuscules."""
    result = clean_tweet("HELLO World")
    assert result == "hello world"


def test_clean_tweet_handles_empty():
    """Vérifie le comportement avec une chaîne vide."""
    assert clean_tweet("") == ""
    assert clean_tweet(None) == ""


def test_clean_tweet_handles_nan():
    """Vérifie le comportement avec NaN."""
    assert clean_tweet(float('nan')) == ""


def test_clean_tweet_strips_whitespace():
    """Vérifie la suppression des espaces multiples."""
    result = clean_tweet("hello   world   test")
    assert result == "hello world test"