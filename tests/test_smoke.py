# -*- coding: utf-8 -*-
"""
Smoke test: Does CERT SDK install and run?

This is the five-minute test - if this fails, nothing else matters.
"""
import subprocess
import sys


def test_import_works():
    """Can we import cert after installation?"""
    import cert
    assert cert is not None


def test_basic_api_available():
    """Are the basic APIs actually importable?"""
    from cert import create_provider
    from cert.core.metrics import behavioral_consistency
    from cert.models import ModelRegistry

    assert create_provider is not None
    assert behavioral_consistency is not None
    assert ModelRegistry is not None


def test_list_models():
    """Can we list models without API keys?"""
    from cert.models import ModelRegistry

    models = ModelRegistry.list_models()
    assert len(models) > 0

    # Check we have the baselines from the paper
    model_ids = [m.model_id for m in models]
    assert "gpt-4o" in model_ids
    assert "claude-3-5-haiku-20241022" in model_ids


def test_metrics_work_without_api():
    """Can we calculate metrics with fake data?"""
    import numpy as np
    from cert.core.metrics import (
        behavioral_consistency,
        coordination_effect,
        pipeline_health_score,
    )

    # Fake semantic distances
    distances = np.array([0.5, 0.6, 0.55, 0.58, 0.52])
    consistency = behavioral_consistency(distances)
    assert 0.0 <= consistency <= 1.0

    # Fake performance scores
    coordinated = 0.8
    independent = np.array([0.6, 0.7])
    gamma = coordination_effect(coordinated, independent)
    assert gamma > 0

    # Fake health score
    health = pipeline_health_score(
        epsilon=0.1,
        gamma_mean=1.2,
        observability_coverage=0.9
    )
    assert 0.0 <= health <= 1.0


if __name__ == "__main__":
    """Run smoke test manually without pytest."""
    print("CERT SDK Smoke Test")
    print("=" * 70)

    tests = [
        ("Import works", test_import_works),
        ("Basic API available", test_basic_api_available),
        ("List models", test_list_models),
        ("Metrics work", test_metrics_work_without_api),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
