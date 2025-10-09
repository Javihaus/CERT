"""
CERT SDK - Coordination Error and Risk Tracking for Production LLM Systems

A production-ready observability framework for multi-agent LLM systems.
"""

__version__ = "0.1.0"

from cert.core.metrics import (
    behavioral_consistency,
    coordination_effect,
    prediction_error,
    pipeline_health_score,
)
from cert.models import ModelRegistry, ModelBaseline, get_model_baseline
from cert.utils import print_models, list_models, get_model_info
from cert.providers import create_provider
from cert.measurements import measure_consistency, measure_performance, measure_agent

# Note: ObservabilityTracker and Pipeline will be available when implemented
# from cert.core.observability import ObservabilityTracker
# from cert.core.pipeline import Pipeline, PipelineConfig

__all__ = [
    "behavioral_consistency",
    "coordination_effect",
    "prediction_error",
    "pipeline_health_score",
    "ModelRegistry",
    "ModelBaseline",
    "get_model_baseline",
    "print_models",
    "list_models",
    "get_model_info",
    "create_provider",
    "measure_consistency",
    "measure_performance",
    "measure_agent",
    # "ObservabilityTracker",
    # "Pipeline",
    # "PipelineConfig",
]
