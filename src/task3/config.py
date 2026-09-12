"""Centralized configuration for local runs and containers."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    environment: str = os.getenv("APP_ENV", "development")
    model_dir: Path = _path_from_env(
        "MODEL_DIR", Path("tasks/task-02-tables-to-notebooks/artifacts/models")
    )
    log_dir: Path = _path_from_env("LOG_DIR", Path("logs"))
    model_version: str = os.getenv("MODEL_VERSION", "task-2-random-forest")
    prediction_threshold: float = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "1000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def model_path(self) -> Path:
        return self.model_dir / "trained_model.joblib"

    @property
    def preprocessor_path(self) -> Path:
        return self.model_dir / "preprocessor.joblib"

    @property
    def category_path(self) -> Path:
        return self.model_dir / "main_product_category_top_values.json"

    @property
    def feature_list_path(self) -> Path:
        return self.model_dir.parent / "reports" / "feature_list.json"


settings = Settings()