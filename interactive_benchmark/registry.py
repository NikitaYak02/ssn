from __future__ import annotations

from pathlib import Path

from .adapters import ExternalSubprocessAdapter, MockGeometryAdapter
from .contracts import MethodAdapter


DEFAULT_EXTERNAL_METHODS = (
    "interformer",
    "seem",
    "semantic_sam",
    "segnext",
    "iseg",
)


def manifests_dir() -> Path:
    return Path(__file__).resolve().parent / "manifests"


def create_adapter(method_id: str) -> MethodAdapter:
    method_id = str(method_id).strip().lower()
    if method_id == "mock_click":
        return MockGeometryAdapter("mock_click", "Mock Click Adapter", "point")
    if method_id == "mock_line":
        return MockGeometryAdapter("mock_line", "Mock Line Adapter", "line")
    if method_id == "mock_scribble":
        return MockGeometryAdapter("mock_scribble", "Mock Scribble Adapter", "scribble")

    manifest_path = manifests_dir() / f"{method_id}.json"
    if manifest_path.exists():
        return ExternalSubprocessAdapter(manifest_path)
    raise KeyError(f"Unknown interactive benchmark method: {method_id}")
