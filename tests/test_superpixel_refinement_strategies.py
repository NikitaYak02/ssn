import numpy as np

from evaluate_superpixel_postprocessing import superpixel_postprocess_strategy
from superpixel_refinement_strategies import (
    SAFE_DEFAULT_STRATEGY_ID,
    SuperpixelRefinementStrategy,
    generate_novel_refinement_strategies,
    generate_safe_refinement_strategies,
    named_strategy_catalog,
)


def test_generate_novel_refinement_strategies_returns_100_unique_items():
    strategies = generate_novel_refinement_strategies(limit=100)

    assert len(strategies) == 100
    assert len({strategy.strategy_id for strategy in strategies}) == 100
    assert all(strategy.strategy_id.startswith("novel_") for strategy in strategies)


def test_low_confidence_strategy_only_overwrites_uncertain_pixels():
    superpixels = np.array([[0, 0, 1, 1]], dtype=np.int32)
    logits = np.array(
        [
            [[5.0, 0.3, 0.2, 4.0]],
            [[0.1, 0.4, 4.0, 0.1]],
        ],
        dtype=np.float32,
    )
    strategy = SuperpixelRefinementStrategy(
        strategy_id="test_lowpix",
        description="Synthetic low-confidence overwrite check",
        family="test",
        aggregate_mode="mean_proba",
        overwrite_policy="low_pixel_conf",
        pixel_confidence_threshold=0.60,
    )

    pred = superpixel_postprocess_strategy(
        logits_np=logits,
        superpixels=superpixels,
        strategy=strategy,
    )

    assert pred.shape == superpixels.shape
    assert pred[0, 0] == 0
    assert pred[0, 1] == 0
    assert pred[0, 2] == 1
    assert pred[0, 3] == 0


def test_safe_guard_preserves_baseline_when_changed_region_is_too_confident():
    superpixels = np.array([[0, 0, 0, 0]], dtype=np.int32)
    logits = np.array(
        [
            [[4.0, 3.2, 3.0, 2.8]],
            [[1.2, 2.7, 2.9, 3.1]],
        ],
        dtype=np.float32,
    )
    strategy = SuperpixelRefinementStrategy(
        strategy_id="test_guard",
        description="Guard should reject wide/high-confidence edits",
        family="test",
        aggregate_mode="mean_proba",
        overwrite_policy="disagree_low_pixel_conf",
        pixel_confidence_threshold=0.80,
        min_superpixel_margin=0.01,
        max_change_fraction=0.25,
        max_changed_mean_confidence=0.55,
    )

    pred = superpixel_postprocess_strategy(
        logits_np=logits,
        superpixels=superpixels,
        strategy=strategy,
    )

    baseline = logits.argmax(axis=0).astype(np.int32)
    assert np.array_equal(pred, baseline)


def test_generate_safe_refinement_strategies_has_expected_family():
    strategies = generate_safe_refinement_strategies()

    assert strategies
    assert any("safe_non_degrading" in strategy.strategy_id for strategy in strategies)


def test_named_catalog_exposes_safe_default_alias():
    catalog = named_strategy_catalog()

    assert "safe_non_degrading" in catalog
    assert catalog["safe_non_degrading"].strategy_id == SAFE_DEFAULT_STRATEGY_ID
