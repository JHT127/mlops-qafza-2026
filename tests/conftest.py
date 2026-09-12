import pytest

from task3.artifacts import load_artifacts
from task3.config import settings
from task3.pipeline import InferencePipeline


@pytest.fixture
def loaded_pipeline():
    artifacts = load_artifacts(settings.model_dir, settings.feature_list_path, settings.model_version)
    return InferencePipeline(artifacts)