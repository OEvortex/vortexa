"""Tests for path-based scoring."""

from vortexa.search.path_scorer import path_score, path_retrieve, filename_features


def test_path_score_exact():
    score = path_score("indexer", "src/vortexa/core/indexer.py")
    assert score > 0.5


def test_path_score_partial():
    score = path_score("index", "src/vortexa/core/indexer.py")
    assert score > 0.0


def test_path_score_no_match():
    score = path_score("banana", "src/vortexa/core/indexer.py")
    assert score == 0.0


def test_path_retrieve():
    files = [
        "src/vortexa/core/indexer.py",
        "src/vortexa/core/types.py",
        "src/vortexa/search/search.py",
        "src/vortexa/search/ranking.py",
    ]
    results = path_retrieve("indexer", files, top_k=3)
    assert len(results) >= 1
    best_path, best_score = results[0]
    assert "indexer" in best_path


def test_filename_features():
    features = filename_features("Fix OAuth bug in auth_service.py")
    assert "auth_service" in features or "oauth" in features
