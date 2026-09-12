import numpy as np
import pytest

from task3.artifacts import ModelArtifacts, load_artifacts
from task3.config import settings
from task3.pipeline import InferencePipeline


class FakePreprocessor:
    def transform(self, frame):
        return np.zeros((len(frame), 2))


class FakeModel:
    classes_ = np.array([0, 1])
    n_features_in_ = 2

    def predict_proba(self, frame):
        return np.tile([0.8, 0.2], (len(frame), 1))


@pytest.fixture
def loaded_pipeline():
    if settings.model_path.exists() and settings.feature_list_path.exists():
        artifacts = load_artifacts(
            settings.model_dir, settings.feature_list_path, settings.model_version
        )
    else:
        artifacts = ModelArtifacts(
            model=FakeModel(),
            preprocessor=FakePreprocessor(),
            top_categories=("health_beauty",),
            feature_list=("feature_1", "feature_2"),
            version="test-model",
        )
    return InferencePipeline(artifacts)
