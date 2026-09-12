"""Load and validate the fitted Task 2 artifacts once per process."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib


class ArtifactError(RuntimeError):
    """Raised when the configured model artifacts are unavailable or incompatible."""


@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    preprocessor: object
    top_categories: tuple[str, ...]
    feature_list: tuple[str, ...]
    version: str


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(f"Could not read artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Invalid JSON artifact: {path}") from exc


def _load_metadata(
    model_dir: Path, feature_list_path: Path
) -> tuple[object, tuple[str, ...], tuple[str, ...]]:
    paths = {
        "preprocessor": model_dir / "preprocessor.joblib",
        "categories": model_dir / "main_product_category_top_values.json",
        "feature_list": feature_list_path,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ArtifactError("Missing model metadata: " + ", ".join(missing))
    return (
        joblib.load(paths["preprocessor"]),
        tuple(_load_json(paths["categories"])),
        tuple(_load_json(paths["feature_list"])),
    )


def load_artifacts(model_dir: Path, feature_list_path: Path, version: str) -> ModelArtifacts:
    paths = {
        "model": model_dir / "trained_model.joblib",
        "preprocessor": model_dir / "preprocessor.joblib",
        "categories": model_dir / "main_product_category_top_values.json",
        "feature_list": feature_list_path,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ArtifactError("Missing model artifacts: " + ", ".join(missing))

    model = joblib.load(paths["model"])
    preprocessor = joblib.load(paths["preprocessor"])
    top_categories = tuple(_load_json(paths["categories"]))
    feature_list = tuple(_load_json(paths["feature_list"]))

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features != len(feature_list):
        raise ArtifactError(
            f"Model expects {expected_features} features, but feature list has "
            f"{len(feature_list)} entries"
        )

    return ModelArtifacts(model, preprocessor, top_categories, feature_list, version)


def load_registry_artifacts(
    model_dir: Path,
    feature_list_path: Path,
    version: str,
    model_uri: str,
    tracking_uri: str = "",
) -> ModelArtifacts:
    """Load the model from MLflow and fitted metadata from the artifact store."""
    try:
        import mlflow
        import mlflow.sklearn

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise ArtifactError(f"Could not load registered model {model_uri}: {exc}") from exc

    preprocessor, top_categories, feature_list = _load_metadata(model_dir, feature_list_path)
    if getattr(model, "n_features_in_", None) != len(feature_list):
        raise ArtifactError("Registered model and feature list have incompatible dimensions")
    return ModelArtifacts(model, preprocessor, top_categories, feature_list, version)
