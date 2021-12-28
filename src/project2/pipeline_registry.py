"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.data_engineering.pipeline import create_pipeline as de_create_pipeline
from .pipelines.data_science.pipeline import create_pipeline as ds_create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """


    data_engineering_pipeline = de_create_pipeline()
    data_science_pipeline = ds_create_pipeline()

    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }
