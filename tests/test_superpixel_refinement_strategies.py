import numpy as np

from evaluate_superpixel_postprocessing import superpixel_postprocess_strategy
from superpixel_refinement_strategies import (
    SuperpixelRefinementStrategy,
    generate_novel_refinement_strategies,
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
