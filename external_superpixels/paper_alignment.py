from __future__ import annotations

from pathlib import Path


REPO_SUPERPIXEL_METHODS = [
    "slic",
    "felzenszwalb",
    "watershed",
    "ssn",
    "deep_slic",
    "cnn_rim",
    "sp_fcn",
    "sin",
    "rethink_unsup",
]

# Variants exposed by the official SPAM codepath (`SPAM/train.py --type_model`).
SPAM_TRAINABLE_VARIANTS = [
    "spam_ssn",
    "spam_resnet50",
    "spam_resnet101",
    "spam_mobilenetv3",
]

# Conservative overlap categories:
# - exact_overlap: same method family appears explicitly in paper/repo.
# - architecture_overlap: SPAM itself is built on top of SSN.
# - lineage_overlap: likely same paper lineage / acronym family, but our
#   implementation is an in-repo approximation rather than upstream code.
EXACT_PAPER_METHOD_OVERLAPS = {
    "slic": "SLIC is explicitly discussed in the SPAM paper as the classic regular superpixel baseline.",
    "ssn": "The official SPAM repo states it is based on the pytorch implementation of SSN.",
}

ARCHITECTURE_OVERLAPS = {
    "ssn": "SPAM inherits the SSN soft clustering core; this is architectural overlap even when the method name differs.",
}

LINEAGE_OVERLAPS = {
    "sp_fcn": (
        "Likely lineage overlap with the SFCN/SSFCN family cited by SPAM; "
        "our in-repo implementation is not the official upstream model."
    ),
    "sin": (
        "Likely lineage overlap with the non-iterative superpixel family "
        "cited by SPAM; our in-repo implementation is an internal approximation."
    ),
}


def compute_superpixel_anything_overlap(
    repo_methods: list[str] | None = None,
) -> dict[str, object]:
    methods = list(repo_methods or REPO_SUPERPIXEL_METHODS)
    method_set = set(methods)
    exact_overlap = sorted([name for name in methods if name in EXACT_PAPER_METHOD_OVERLAPS])
    architecture_overlap = sorted([name for name in methods if name in ARCHITECTURE_OVERLAPS])
    lineage_overlap = sorted([name for name in methods if name in LINEAGE_OVERLAPS])
    spam_variant_overlap = sorted(method_set.intersection(SPAM_TRAINABLE_VARIANTS))
    return {
        "repo_methods": methods,
        "spam_trainable_variants": list(SPAM_TRAINABLE_VARIANTS),
        "exact_overlap": exact_overlap,
        "architecture_overlap": architecture_overlap,
        "lineage_overlap": lineage_overlap,
        "spam_variant_overlap": spam_variant_overlap,
        "notes": {
            "exact_overlap": {name: EXACT_PAPER_METHOD_OVERLAPS[name] for name in exact_overlap},
            "architecture_overlap": {
                name: ARCHITECTURE_OVERLAPS[name] for name in architecture_overlap
            },
            "lineage_overlap": {name: LINEAGE_OVERLAPS[name] for name in lineage_overlap},
        },
    }


def build_superpixel_anything_overlap_report(
    *,
    repo_methods: list[str] | None = None,
    paper_url: str = "https://arxiv.org/abs/2509.12791",
    repo_url: str = "https://github.com/waldo-j/spam",
) -> str:
    overlap = compute_superpixel_anything_overlap(repo_methods)
    exact = overlap["exact_overlap"]
    architecture = overlap["architecture_overlap"]
    lineage = overlap["lineage_overlap"]
    spam_variant_overlap = overlap["spam_variant_overlap"]
    notes = overlap["notes"]

    lines = [
        "# Superpixel Anything Overlap Check",
        "",
        f"- Paper: {paper_url}",
        f"- Official repo: {repo_url}",
        "",
        "## Trainable SPAM variants exposed in the official code",
        "",
    ]
    for name in overlap["spam_trainable_variants"]:
        lines.append(f"- `{name}`")

    lines.extend(
        [
            "",
            "## Check Result",
            "",
            "Проверка на \"совпадений нет\" не проходит в строгом виде.",
            "",
            "### Exact overlaps with methods already present in this repository",
            "",
        ]
    )
    if exact:
        for name in exact:
            lines.append(f"- `{name}`: {notes['exact_overlap'][name]}")
    else:
        lines.append("- Exact overlaps not found.")

    lines.extend(
        [
            "",
            "### Architectural overlaps",
            "",
        ]
    )
    if architecture:
        for name in architecture:
            lines.append(f"- `{name}`: {notes['architecture_overlap'][name]}")
    else:
        lines.append("- Architectural overlaps not found.")

    lines.extend(
        [
            "",
            "### Lineage-level / likely family overlaps",
            "",
        ]
    )
    if lineage:
        for name in lineage:
            lines.append(f"- `{name}`: {notes['lineage_overlap'][name]}")
    else:
        lines.append("- No likely family overlaps were flagged.")

    lines.extend(
        [
            "",
            "### Exact overlap with trainable SPAM variants",
            "",
        ]
    )
    if spam_variant_overlap:
        for name in spam_variant_overlap:
            lines.append(f"- `{name}`")
    else:
        lines.append(
            "- No exact method-id overlap with the new SPAM trainable variants "
            "(`spam_ssn`, `spam_resnet50`, `spam_resnet101`, `spam_mobilenetv3`)."
        )

    lines.extend(
        [
            "",
            "## Practical conclusion",
            "",
            "- Adding SPAM training support is still useful.",
            "- But it should not be presented as fully disjoint from prior repo work, because `SSN` and `SLIC` already overlap with SPAM paper context, and `SP-FCN` / `SIN` are close family-level matches.",
            "",
        ]
    )
    return "\n".join(lines)


def default_overlap_report_path(root: Path) -> Path:
    return root / "reports" / "superpixel_anything_overlap.md"
