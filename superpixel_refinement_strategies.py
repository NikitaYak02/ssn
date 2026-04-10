from __future__ import annotations

from dataclasses import asdict, dataclass


AGGREGATE_MODES = {
    "mean_proba",
    "majority_argmax",
    "logit_mean",
    "confidence_weighted_mean",
    "margin_weighted_mean",
    "entropy_weighted_mean",
}

OVERWRITE_POLICIES = {
    "all",
    "low_pixel_conf",
    "disagree_low_pixel_conf",
    "high_sp_conf",
    "changed_high_sp_conf",
}

CLEANUP_MODES = {"none", "simple", "conservative"}
SAFE_DEFAULT_STRATEGY_ID = "safe_mean_t075_safe_non_degrading_v2"


@dataclass(frozen=True)
class SuperpixelRefinementStrategy:
    strategy_id: str
    description: str
    family: str
    aggregate_mode: str = "mean_proba"
    overwrite_policy: str = "all"
    pixel_confidence_threshold: float = 0.75
    superpixel_confidence_threshold: float = 0.75
    prior_power: float = 0.0
    temperature: float = 1.0
    weight_power: float = 1.0
    graph_steps: int = 0
    graph_alpha: float = 0.0
    cleanup_mode: str = "none"
    small_component_superpixels: int = 3
    neighbor_ratio_threshold: float = 0.6
    min_superpixel_margin: float = 0.0
    max_change_fraction: float = 1.0
    max_changed_mean_confidence: float = 1.0
    enforce_superpixel_confidence_guard: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_strategy(strategy: SuperpixelRefinementStrategy) -> SuperpixelRefinementStrategy:
    if strategy.aggregate_mode not in AGGREGATE_MODES:
        raise ValueError(f"Unknown aggregate_mode: {strategy.aggregate_mode}")
    if strategy.overwrite_policy not in OVERWRITE_POLICIES:
        raise ValueError(f"Unknown overwrite_policy: {strategy.overwrite_policy}")
    if strategy.cleanup_mode not in CLEANUP_MODES:
        raise ValueError(f"Unknown cleanup_mode: {strategy.cleanup_mode}")
    if strategy.temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if strategy.weight_power <= 0.0:
        raise ValueError("weight_power must be > 0")
    if strategy.graph_steps < 0:
        raise ValueError("graph_steps must be >= 0")
    if not (0.0 <= strategy.graph_alpha <= 1.0):
        raise ValueError("graph_alpha must be in [0, 1]")
    if strategy.small_component_superpixels < 1:
        raise ValueError("small_component_superpixels must be >= 1")
    if strategy.min_superpixel_margin < 0.0:
        raise ValueError("min_superpixel_margin must be >= 0")
    if not (0.0 < strategy.max_change_fraction <= 1.0):
        raise ValueError("max_change_fraction must be in (0, 1]")
    if not (0.0 < strategy.max_changed_mean_confidence <= 1.0):
        raise ValueError("max_changed_mean_confidence must be in (0, 1]")
    return strategy


def build_legacy_strategy(
    vote_mode: str,
    *,
    confidence_threshold: float = 0.75,
    prior_power: float = 0.5,
    small_component_superpixels: int = 3,
    hybrid_neighbor_ratio: float = 0.6,
) -> SuperpixelRefinementStrategy:
    table = {
        "mean_proba": SuperpixelRefinementStrategy(
            strategy_id="legacy_mean_proba",
            description="Legacy mean probability per superpixel.",
            family="legacy",
            aggregate_mode="mean_proba",
        ),
        "majority_argmax": SuperpixelRefinementStrategy(
            strategy_id="legacy_majority_argmax",
            description="Legacy majority vote over pixel argmax labels.",
            family="legacy",
            aggregate_mode="majority_argmax",
        ),
        "confidence_gated_mean_proba": SuperpixelRefinementStrategy(
            strategy_id="legacy_confidence_gated_mean_proba",
            description="Overwrite only highly confident superpixels.",
            family="legacy",
            aggregate_mode="mean_proba",
            overwrite_policy="high_sp_conf",
            superpixel_confidence_threshold=float(confidence_threshold),
        ),
        "low_confidence_mean_proba": SuperpixelRefinementStrategy(
            strategy_id="legacy_low_confidence_mean_proba",
            description="Overwrite only low-confidence pixels.",
            family="legacy",
            aggregate_mode="mean_proba",
            overwrite_policy="low_pixel_conf",
            pixel_confidence_threshold=float(confidence_threshold),
        ),
        "prior_corrected_mean_proba": SuperpixelRefinementStrategy(
            strategy_id="legacy_prior_corrected_mean_proba",
            description="Mean probabilities with image-level prior correction.",
            family="legacy",
            aggregate_mode="mean_proba",
            prior_power=float(prior_power),
        ),
        "small_region_cleanup": SuperpixelRefinementStrategy(
            strategy_id="legacy_small_region_cleanup",
            description="Mean probabilities plus connected-component cleanup.",
            family="legacy",
            aggregate_mode="mean_proba",
            cleanup_mode="simple",
            small_component_superpixels=int(small_component_superpixels),
        ),
        "hybrid_conservative": SuperpixelRefinementStrategy(
            strategy_id="legacy_hybrid_conservative",
            description="Low-confidence overwrite plus conservative cleanup.",
            family="legacy",
            aggregate_mode="mean_proba",
            overwrite_policy="low_pixel_conf",
            pixel_confidence_threshold=float(confidence_threshold),
            cleanup_mode="conservative",
            small_component_superpixels=int(small_component_superpixels),
            neighbor_ratio_threshold=float(hybrid_neighbor_ratio),
        ),
    }
    if vote_mode not in table:
        raise KeyError(f"Unknown legacy vote mode: {vote_mode}")
    return _validate_strategy(table[vote_mode])


def legacy_strategy_catalog(
    *,
    confidence_threshold: float = 0.75,
    prior_power: float = 0.5,
    small_component_superpixels: int = 3,
    hybrid_neighbor_ratio: float = 0.6,
) -> list[SuperpixelRefinementStrategy]:
    return [
        build_legacy_strategy(
            vote_mode,
            confidence_threshold=confidence_threshold,
            prior_power=prior_power,
            small_component_superpixels=small_component_superpixels,
            hybrid_neighbor_ratio=hybrid_neighbor_ratio,
        )
        for vote_mode in (
            "mean_proba",
            "majority_argmax",
            "confidence_gated_mean_proba",
            "low_confidence_mean_proba",
            "prior_corrected_mean_proba",
            "small_region_cleanup",
            "hybrid_conservative",
        )
    ]


def generate_novel_refinement_strategies(limit: int = 100) -> list[SuperpixelRefinementStrategy]:
    if limit <= 0:
        return []

    aggregate_recipes = [
        {
            "family": "mean_base",
            "aggregate_mode": "mean_proba",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 1.0,
        },
        {
            "family": "mean_t085",
            "aggregate_mode": "mean_proba",
            "temperature": 0.85,
            "prior_power": 0.0,
            "weight_power": 1.0,
        },
        {
            "family": "mean_t075",
            "aggregate_mode": "mean_proba",
            "temperature": 0.75,
            "prior_power": 0.0,
            "weight_power": 1.0,
        },
        {
            "family": "mean_p025",
            "aggregate_mode": "mean_proba",
            "temperature": 1.0,
            "prior_power": 0.25,
            "weight_power": 1.0,
        },
        {
            "family": "mean_p050",
            "aggregate_mode": "mean_proba",
            "temperature": 1.0,
            "prior_power": 0.50,
            "weight_power": 1.0,
        },
        {
            "family": "mean_t085_p025",
            "aggregate_mode": "mean_proba",
            "temperature": 0.85,
            "prior_power": 0.25,
            "weight_power": 1.0,
        },
        {
            "family": "mean_t075_p025",
            "aggregate_mode": "mean_proba",
            "temperature": 0.75,
            "prior_power": 0.25,
            "weight_power": 1.0,
        },
        {
            "family": "logit_mean",
            "aggregate_mode": "logit_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 1.0,
        },
        {
            "family": "logit_mean_p025",
            "aggregate_mode": "logit_mean",
            "temperature": 1.0,
            "prior_power": 0.25,
            "weight_power": 1.0,
        },
        {
            "family": "confw_p10",
            "aggregate_mode": "confidence_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 1.0,
        },
        {
            "family": "confw_p15",
            "aggregate_mode": "confidence_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 1.5,
        },
        {
            "family": "confw_p20",
            "aggregate_mode": "confidence_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 2.0,
        },
        {
            "family": "confw_p20_t085",
            "aggregate_mode": "confidence_weighted_mean",
            "temperature": 0.85,
            "prior_power": 0.0,
            "weight_power": 2.0,
        },
        {
            "family": "margin_p10",
            "aggregate_mode": "margin_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 1.0,
        },
        {
            "family": "margin_p15",
            "aggregate_mode": "margin_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 1.5,
        },
        {
            "family": "margin_p20",
            "aggregate_mode": "margin_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 2.0,
        },
        {
            "family": "margin_p20_t085",
            "aggregate_mode": "margin_weighted_mean",
            "temperature": 0.85,
            "prior_power": 0.0,
            "weight_power": 2.0,
        },
        {
            "family": "entropy_p10",
            "aggregate_mode": "entropy_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 1.0,
        },
        {
            "family": "entropy_p20",
            "aggregate_mode": "entropy_weighted_mean",
            "temperature": 1.0,
            "prior_power": 0.0,
            "weight_power": 2.0,
        },
        {
            "family": "entropy_p20_t085",
            "aggregate_mode": "entropy_weighted_mean",
            "temperature": 0.85,
            "prior_power": 0.0,
            "weight_power": 2.0,
        },
    ]

    refinement_profiles = [
        {
            "profile": "full_direct",
            "description": "Broadcast refined superpixel label to all pixels.",
            "overwrite_policy": "all",
            "pixel_confidence_threshold": 0.75,
            "superpixel_confidence_threshold": 0.75,
            "graph_steps": 0,
            "graph_alpha": 0.0,
            "cleanup_mode": "none",
            "small_component_superpixels": 3,
            "neighbor_ratio_threshold": 0.6,
        },
        {
            "profile": "lowpix_060",
            "description": "Overwrite only low-confidence pixels.",
            "overwrite_policy": "low_pixel_conf",
            "pixel_confidence_threshold": 0.60,
            "superpixel_confidence_threshold": 0.75,
            "graph_steps": 0,
            "graph_alpha": 0.0,
            "cleanup_mode": "none",
            "small_component_superpixels": 3,
            "neighbor_ratio_threshold": 0.6,
        },
        {
            "profile": "lowpix_075_graph1",
            "description": "Low-confidence overwrite with one graph-smoothing pass.",
            "overwrite_policy": "low_pixel_conf",
            "pixel_confidence_threshold": 0.75,
            "superpixel_confidence_threshold": 0.75,
            "graph_steps": 1,
            "graph_alpha": 0.20,
            "cleanup_mode": "none",
            "small_component_superpixels": 3,
            "neighbor_ratio_threshold": 0.6,
        },
        {
            "profile": "disagree_070_graph1_simple2",
            "description": "Patch only disagreeing low-confidence pixels and merge tiny islands.",
            "overwrite_policy": "disagree_low_pixel_conf",
            "pixel_confidence_threshold": 0.70,
            "superpixel_confidence_threshold": 0.75,
            "graph_steps": 1,
            "graph_alpha": 0.35,
            "cleanup_mode": "simple",
            "small_component_superpixels": 2,
            "neighbor_ratio_threshold": 0.6,
        },
        {
            "profile": "highsp_065_graph2_cons3",
            "description": "Apply only strong superpixels after graph smoothing and conservative cleanup.",
            "overwrite_policy": "high_sp_conf",
            "pixel_confidence_threshold": 0.75,
            "superpixel_confidence_threshold": 0.65,
            "graph_steps": 2,
            "graph_alpha": 0.25,
            "cleanup_mode": "conservative",
            "small_component_superpixels": 3,
            "neighbor_ratio_threshold": 0.6,
        },
    ]

    out: list[SuperpixelRefinementStrategy] = []
    seen: set[str] = set()
    for agg_idx, aggregate in enumerate(aggregate_recipes, start=1):
        for profile_idx, profile in enumerate(refinement_profiles, start=1):
            strategy = SuperpixelRefinementStrategy(
                strategy_id=f"novel_{agg_idx:02d}_{profile_idx:02d}",
                description=(
                    f"{aggregate['family']} + {profile['profile']}: "
                    f"{profile['description']}"
                ),
                family=str(aggregate["family"]),
                aggregate_mode=str(aggregate["aggregate_mode"]),
                overwrite_policy=str(profile["overwrite_policy"]),
                pixel_confidence_threshold=float(profile["pixel_confidence_threshold"]),
                superpixel_confidence_threshold=float(profile["superpixel_confidence_threshold"]),
                prior_power=float(aggregate["prior_power"]),
                temperature=float(aggregate["temperature"]),
                weight_power=float(aggregate["weight_power"]),
                graph_steps=int(profile["graph_steps"]),
                graph_alpha=float(profile["graph_alpha"]),
                cleanup_mode=str(profile["cleanup_mode"]),
                small_component_superpixels=int(profile["small_component_superpixels"]),
                neighbor_ratio_threshold=float(profile["neighbor_ratio_threshold"]),
            )
            strategy = _validate_strategy(strategy)
            if strategy.strategy_id in seen:
                continue
            out.append(strategy)
            seen.add(strategy.strategy_id)
            if len(out) >= limit:
                return out
    return out


def generate_safe_refinement_strategies() -> list[SuperpixelRefinementStrategy]:
    aggregates = [
        ("safe_mean_t075", "mean_proba", 0.75, 0.0, 1.0),
        ("safe_mean_t085", "mean_proba", 0.85, 0.0, 1.0),
        ("safe_confw_p10", "confidence_weighted_mean", 1.0, 0.0, 1.0),
        ("safe_confw_p20_t085", "confidence_weighted_mean", 0.85, 0.0, 2.0),
        ("safe_margin_p15", "margin_weighted_mean", 1.0, 0.0, 1.5),
    ]
    profiles = [
        (
            "safe_non_degrading_v1",
            "Strict disagreement-only updates with tight baseline-preserving guard.",
            0.45,
            1,
            0.15,
            "none",
            2,
            0.10,
            0.10,
            0.45,
            0.88,
        ),
        (
            "safe_non_degrading_v2",
            "Disagreement-only updates with moderate guard and very limited footprint.",
            0.50,
            1,
            0.20,
            "none",
            2,
            0.08,
            0.15,
            0.48,
            0.86,
        ),
        (
            "safe_non_degrading_v3",
            "Conservative disagreement updates with tiny-island cleanup only under strong evidence.",
            0.50,
            1,
            0.25,
            "simple",
            2,
            0.12,
            0.12,
            0.50,
            0.90,
        ),
    ]
    out: list[SuperpixelRefinementStrategy] = []
    for family, aggregate_mode, temperature, prior_power, weight_power in aggregates:
        for (
            suffix,
            description,
            pixel_thr,
            graph_steps,
            graph_alpha,
            cleanup_mode,
            small_comp,
            min_margin,
            max_frac,
            max_mean_conf,
            sp_conf_thr,
        ) in profiles:
            out.append(
                _validate_strategy(
                    SuperpixelRefinementStrategy(
                        strategy_id=f"{family}_{suffix}",
                        description=description,
                        family=family,
                        aggregate_mode=aggregate_mode,
                        overwrite_policy="disagree_low_pixel_conf",
                        pixel_confidence_threshold=float(pixel_thr),
                        superpixel_confidence_threshold=float(sp_conf_thr),
                        prior_power=float(prior_power),
                        temperature=float(temperature),
                        weight_power=float(weight_power),
                        graph_steps=int(graph_steps),
                        graph_alpha=float(graph_alpha),
                        cleanup_mode=cleanup_mode,
                        small_component_superpixels=int(small_comp),
                        neighbor_ratio_threshold=0.6,
                        min_superpixel_margin=float(min_margin),
                        max_change_fraction=float(max_frac),
                        max_changed_mean_confidence=float(max_mean_conf),
                        enforce_superpixel_confidence_guard=True,
                    )
                )
            )
    return out


def named_strategy_catalog(
    *,
    confidence_threshold: float = 0.75,
    prior_power: float = 0.5,
    small_component_superpixels: int = 3,
    hybrid_neighbor_ratio: float = 0.6,
    include_legacy: bool = True,
    novel_limit: int = 100,
) -> dict[str, SuperpixelRefinementStrategy]:
    strategies: list[SuperpixelRefinementStrategy] = []
    if include_legacy:
        strategies.extend(
            legacy_strategy_catalog(
                confidence_threshold=confidence_threshold,
                prior_power=prior_power,
                small_component_superpixels=small_component_superpixels,
                hybrid_neighbor_ratio=hybrid_neighbor_ratio,
            )
        )
    strategies.extend(generate_safe_refinement_strategies())
    strategies.extend(generate_novel_refinement_strategies(limit=novel_limit))
    catalog = {strategy.strategy_id: strategy for strategy in strategies}
    if SAFE_DEFAULT_STRATEGY_ID in catalog:
        catalog["safe_non_degrading"] = catalog[SAFE_DEFAULT_STRATEGY_ID]
    return catalog
