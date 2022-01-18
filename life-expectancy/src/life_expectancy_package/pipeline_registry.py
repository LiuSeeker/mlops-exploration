"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.pipeline import create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    main_pipeline = create_pipeline()

    return {"__default__": Pipeline([main_pipeline])}
