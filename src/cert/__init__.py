"""
CERT SDK - Coordination Error and Risk Tracking for Production LLM Systems

A production-ready observability framework for multi-agent LLM systems.
"""

__version__ = "0.1.0"

from cert.core.metrics import (
    behavioral_consistency,
    coordination_effect,
    pipeline_health_score,
    prediction_error,
)
from cert.measurements import (
    measure_agent,
    measure_consistency,
    measure_custom_baseline,
    measure_performance,
)
from cert.models import ModelBaseline, ModelRegistry, get_model_baseline
from cert.providers import create_provider
from cert.utils import get_model_info, list_models, print_models

# Note: ObservabilityTracker and Pipeline will be available when implemented
# from cert.core.observability import ObservabilityTracker
# from cert.core.pipeline import Pipeline, PipelineConfig

__all__ = [
    "ModelBaseline",
    "ModelRegistry",
    "behavioral_consistency",
    "coordination_effect",
    "create_provider",
    "get_model_baseline",
    "get_model_info",
    "list_models",
    "measure_agent",
    "measure_consistency",
    "measure_custom_baseline",
    "measure_performance",
    "pipeline_health_score",
    "prediction_error",
    "print_models",
    # "ObservabilityTracker",
    # "Pipeline",
    # "PipelineConfig",
]
