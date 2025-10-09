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
from cert.core.observability import ObservabilityTracker
from cert.core.pipeline import Pipeline, PipelineConfig

__all__ = [
    "behavioral_consistency",
    "coordination_effect",
    "prediction_error",
    "pipeline_health_score",
    "ObservabilityTracker",
    "Pipeline",
    "PipelineConfig",
]
