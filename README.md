# CERT SDK

[![PyPI](https://img.shields.io/pypi/v/cert-sdk.svg)](https://pypi.org/project/cert-sdk/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/cert-sdk.svg)](https://pypi.org/project/cert-sdk/)
[![Documentation](https://img.shields.io/badge/docs-cert--sdk-blue.svg)](https://cert-sdk.readthedocs.io)

A Python SDK for observability and reliability tracking in multi-model LLM sequential processing.

CERT (Consistency, Effect, and Reliability Tracking) provides production-grade instrumentation for monitoring, debugging, and validating sequential LLM pipelines. Built for teams deploying multi-agent AI systems at scale.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
  - [Sequential Pipeline Measurement](#sequential-pipeline-measurement)
- [Core Concepts](#core-concepts)
  - [Behavioral Consistency](#behavioral-consistency)
  - [Context Propagation Effect](#context-propagation-effect)
  - [Pipeline Health](#pipeline-health)
- [Framework Integrations](#framework-integrations)
  - [LangChain](#langchain)
  - [CrewAI](#crewai)
  - [AutoGen](#autogen)
- [Use Cases](#use-cases)
  - [Pre-Deployment Validation](#pre-deployment-validation)
  - [Model Drift Detection](#model-drift-detection)
  - [Architecture Optimization](#architecture-optimization)
- [API Reference](#api-reference)
  - [Providers](#providers)
  - [Measurement Functions](#measurement-functions)
  - [Pipeline Health Assessment](#pipeline-health-assessment)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
  - [Custom Metrics](#custom-metrics)
  - [Batch Processing](#batch-processing)
  - [Experiment Tracking](#experiment-tracking)
- [Empirical Baselines](#empirical-baselines)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)
- [Citation](#citation)

## Overview

Modern LLM applications increasingly rely on sequential processing through multiple models and agents. CERT provides the observability infrastructure needed to measure, monitor, and maintain reliability in these complex systems.

The SDK offers three core metrics:

- **Behavioral Consistency (C)**: Quantifies output variance for identical inputs, essential for detecting model drift and validating deployment stability
- **Context Propagation Effect (γ)**: Measures how performance changes as context accumulates through sequential processing stages
- **Pipeline Health (H)**: A composite operational metric that enables data-driven deployment decisions

## Installation

Install CERT using pip:

```bash
pip install cert-sdk
```

Or install from source:

```bash
git clone https://github.com/Javihaus/CERT.git
cd CERT
pip install -e .
```

## Quick Start

### Basic Usage

Measure consistency and performance for a single model:

```python
import cert
import asyncio

async def main():
    # Initialize provider
    provider = cert.create_provider(
        api_key="sk-...",
        model_name="gpt-4o"
    )

    # Run measurements
    results = await cert.measure_agent(
        provider,
        n_consistency_trials=10
    )

    # Analyze metrics
    print(f"Consistency Score: {results['consistency']:.3f}")
    print(f"Mean Performance: {results['mean_performance']:.3f}")
    print(f"Pipeline Health: {results['health']:.3f}")

asyncio.run(main())
```

### Sequential Pipeline Measurement

Monitor multi-stage LLM pipelines:

```python
import cert

# Define sequential pipeline
pipeline = cert.Pipeline([
    cert.create_provider(model_name="gpt-4o", api_key="sk-..."),
    cert.create_provider(model_name="claude-3-opus", api_key="sk-ant-..."),
])

# Measure context propagation effects
results = await cert.measure_pipeline(
    pipeline,
    input_samples=test_cases,
    n_trials=5
)

print(f"Context Effect (γ): {results['gamma']:.3f}")
print(f"Stage 1 Consistency: {results['stage_consistency'][0]:.3f}")
print(f"Stage 2 Consistency: {results['stage_consistency'][1]:.3f}")
```

## Core Concepts

### Behavioral Consistency

Consistency measures output variance when processing identical inputs multiple times. High consistency indicates stable, predictable behavior essential for production systems.

```python
# Measure consistency with custom trials
consistency_score = await cert.measure_consistency(
    provider,
    input_prompt="Analyze this data: [...]",
    n_trials=20
)
```

Low consistency scores may indicate:
- Model temperature settings too high
- Non-deterministic processing in pipeline
- Model version drift
- Infrastructure issues affecting model behavior

### Context Propagation Effect

The gamma (γ) metric quantifies how accumulated context affects performance through sequential processing stages. Understanding this effect is critical for optimizing multi-agent architectures.

CERT provides both raw γ and normalized γ_norm values. The normalized form (γ_norm = γ^(1/n)) enables fair comparison across pipelines of different lengths.

```python
# Analyze context effects across stages
effect = await cert.measure_context_effect(
    pipeline,
    baseline_input=simple_task,
    accumulated_context=previous_outputs
)

print(f"Raw gamma: {effect['gamma']:.3f}")
print(f"Normalized gamma: {effect['gamma_norm']:.3f}")
print(f"Pipeline length: {effect['n_agents']}")

# Interpret normalized gamma (per-step context benefit)
if effect['gamma_norm'] > 1.5:
    print("Strong per-step context benefit - sequential architecture recommended")
elif effect['gamma_norm'] > 1.2:
    print("Moderate context propagation - deploy with monitoring")
elif effect['gamma_norm'] > 1.0:
    print("Weak context effect - consider evaluation")
else:
    print("Context degradation - investigate pipeline structure")
```

**Operational Thresholds for γ_norm (Table 9 from paper):**

| γ_norm Range | Interpretation | Recommendation |
|--------------|----------------|----------------|
| > 1.5 | Strong per-step context benefit | Sequential architecture recommended |
| 1.2 - 1.5 | Moderate context propagation | Deploy with monitoring |
| 1.0 - 1.2 | Weak context effect | Consider additional evaluation |
| < 1.0 | Context degradation | Investigate pipeline structure |

**Model Baselines from Paper:**

| Model | 2-Agent γ_norm | 5-Agent γ_norm | Health (5-Agent) |
|-------|----------------|----------------|------------------|
| Claude Haiku 3.5 | 1.209 | 1.685 | 0.43 |
| GPT-4o | 1.250 | 1.578 | 0.49 |
| Grok 3 | 1.275 | 1.617 | 0.47 |
| Gemini 3.5 Pro | 1.066 | 1.521 | 0.62 |

### Pipeline Health

Pipeline Health provides a single composite metric for operational decision-making. The health score uses **normalized γ_norm** (not raw γ) to ensure comparability across different pipeline lengths.

**Health Score Formula (Equation 8):**
```
H = (1 / (1 + ε)) × min(1, γ_norm) × C_obs
```

Where:
- **ε** = Prediction error (relative to baseline with degradation factor α=0.93)
- **γ_norm** = Normalized context propagation effect (γ^(1/n))
- **C_obs** = Observability coverage (fraction of instrumented interactions)

**Baseline Calculation (Equation 6):**
The predicted baseline uses a product baseline with information degradation:
```
P_baseline = (∏ μ_i) × α^(n-1)
```

Where α=0.93 models cumulative information loss through repeated encoding/decoding cycles.

```python
health = await cert.assess_pipeline_health(pipeline)

print(f"Health Score: {health['score']:.2f}")
print(f"Prediction Error: {health['epsilon']:.2f}")
print(f"Normalized Gamma: {health['gamma_norm']:.3f}")
print(f"Observability: {health['observability_coverage']:.2f}")

if health['score'] > 0.8:
    print("Pipeline ready for production")
elif health['score'] > 0.6:
    print("Pipeline acceptable, monitor closely")
else:
    print("Pipeline requires optimization")
    print(f"Issues: {health['issues']}")
```

**Health Score Thresholds:**

| Score Range | Status | Action |
|-------------|--------|--------|
| H > 0.8 | Healthy | Standard monitoring |
| 0.6 < H ≤ 0.8 | Acceptable | Monitor for degradation |
| 0.4 < H ≤ 0.6 | Degraded | Investigate issues |
| H ≤ 0.4 | Critical | Immediate attention required |

## Framework Integrations

CERT provides drop-in instrumentation for popular LLM frameworks.

### LangChain

```python
from langchain.agents import AgentExecutor
from cert.integrations import langchain

# Wrap existing LangChain agent
agent_executor = AgentExecutor(...)
instrumented_agent = langchain.instrument(agent_executor)

# Automatic metrics collection
results = await instrumented_agent.run(query)
metrics = langchain.get_metrics(instrumented_agent)
```

### CrewAI

```python
from crewai import Crew
from cert.integrations import crewai

crew = Crew(agents=[...], tasks=[...])
instrumented_crew = crewai.instrument(crew)

# Monitor crew execution
output = await instrumented_crew.kickoff()
analytics = crewai.get_analytics(instrumented_crew)
```

### AutoGen

```python
from autogen import AssistantAgent
from cert.integrations import autogen

agent = AssistantAgent(...)
instrumented_agent = autogen.instrument(agent)

# Track multi-agent conversations
response = await instrumented_agent.generate_reply(messages)
metrics = autogen.get_conversation_metrics(instrumented_agent)
```

## Use Cases

### Pre-Deployment Validation

Establish baseline metrics before deploying new models or architectures:

```python
# Establish baseline
baseline = await cert.establish_baseline(
    provider,
    test_suite=validation_cases,
    n_trials=10
)

# Compare against new model
candidate = cert.create_provider(model_name="gpt-4o-mini", api_key="sk-...")
comparison = await cert.compare_to_baseline(
    candidate,
    baseline,
    test_suite=validation_cases
)

if comparison['consistency_delta'] > -0.1 and comparison['performance_delta'] > -0.05:
    print("Candidate model approved for deployment")
```

### Model Drift Detection

Monitor production systems for behavioral changes:

```python
import cert.monitoring

# Set up continuous monitoring
monitor = cert.monitoring.create_monitor(
    pipeline,
    check_interval="1h",
    baseline=production_baseline
)

# Alert on significant drift
@monitor.on_drift
def handle_drift(drift_report):
    if drift_report['consistency_change'] < -0.15:
        alert_team(f"Critical consistency drift: {drift_report}")
```

### Architecture Optimization

Quantify improvements from architectural changes:

```python
# Test architectural variations
architectures = [
    simple_sequential_pipeline,
    parallel_processing_pipeline,
    hierarchical_pipeline
]

results = await cert.benchmark_architectures(
    architectures,
    test_suite=benchmark_cases
)

best = max(results, key=lambda r: r['health_score'])
print(f"Optimal architecture: {best['name']}")
print(f"Health score: {best['health_score']:.3f}")
```

## API Reference

### Providers

Create model providers for measurement:

```python
provider = cert.create_provider(
    model_name: str,           # Model identifier (e.g., "gpt-4o", "claude-3-opus")
    api_key: str,              # API authentication key
    temperature: float = 0.7,  # Sampling temperature
    max_tokens: int = 1000,    # Maximum output tokens
    **kwargs                   # Additional provider-specific parameters
)
```

### Measurement Functions

#### `measure_agent()`

Measure consistency and performance for a single agent:

```python
results = await cert.measure_agent(
    provider: Provider,
    n_consistency_trials: int = 10,
    input_samples: List[str] = None,
    metrics: List[str] = ["consistency", "performance"]
) -> Dict[str, float]
```

#### `measure_pipeline()`

Evaluate multi-stage sequential pipelines:

```python
results = await cert.measure_pipeline(
    pipeline: Pipeline,
    input_samples: List[str],
    n_trials: int = 5,
    measure_stages: bool = True
) -> Dict[str, Any]
```

#### `establish_baseline()`

Create baseline measurements for comparison:

```python
baseline = await cert.establish_baseline(
    provider: Provider,
    test_suite: List[TestCase],
    n_trials: int = 10,
    save_path: str = None
) -> Baseline
```

### Pipeline Health Assessment

```python
health = await cert.assess_pipeline_health(
    pipeline: Pipeline,
    baseline: Baseline = None,
    thresholds: Dict[str, float] = None
) -> HealthReport
```

## Configuration

Configure CERT using environment variables or a configuration file:

```python
# Environment variables
export CERT_LOG_LEVEL=INFO
export CERT_METRICS_BACKEND=prometheus
export CERT_CACHE_DIR=/tmp/cert_cache

# Or use configuration file
cert.configure(
    log_level="INFO",
    metrics_backend="prometheus",
    cache_dir="/tmp/cert_cache",
    retry_policy={
        "max_attempts": 3,
        "backoff_factor": 2.0
    }
)
```

## Advanced Features

### Custom Metrics

Define domain-specific metrics:

```python
from cert.metrics import CustomMetric

class TaskAccuracy(CustomMetric):
    def evaluate(self, output: str, expected: str) -> float:
        # Implement custom evaluation logic
        return compute_accuracy(output, expected)

# Use custom metric
results = await cert.measure_agent(
    provider,
    custom_metrics=[TaskAccuracy()]
)
```

### Batch Processing

Efficiently process large evaluation sets:

```python
results = await cert.batch_evaluate(
    provider,
    inputs=large_test_set,
    batch_size=50,
    parallel_workers=5
)
```

### Experiment Tracking

Integrate with experiment tracking systems:

```python
from cert.tracking import MLflowTracker

tracker = MLflowTracker(experiment_name="model-validation")

with tracker.track():
    results = await cert.measure_agent(provider)
    tracker.log_metrics(results)
    tracker.log_artifacts({"baseline": baseline_data})
```

## Empirical Baselines

CERT includes empirical baseline measurements for popular models:

```python
# Access pre-computed baselines
baseline = cert.baselines.get("gpt-4o", task_type="general")
print(f"Expected consistency: {baseline['consistency_range']}")

# Compare against baseline
results = await cert.measure_agent(provider)
delta = results['consistency'] - baseline['consistency_mean']

if abs(delta) > baseline['consistency_std'] * 2:
    print("Warning: Significant deviation from expected behavior")
```

Available baselines:
- GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- Gemini 1.5 Pro, Gemini 1.5 Flash
- Llama 3 70B, Llama 3 8B

## Contributing

We welcome contributions to CERT. Please see our [contributing guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs and requesting features
- Submitting pull requests
- Development setup and testing
- Code style and documentation standards

## License

CERT is released under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- Documentation: [https://cert-sdk.readthedocs.io](https://cert-sdk.readthedocs.io)
- Issues: [https://github.com/Javihaus/CERT/issues](https://github.com/Javihaus/CERT/issues)
- Discussions: [https://github.com/Javihaus/CERT/discussions](https://github.com/Javihaus/CERT/discussions)

## Citation

If you use CERT in your research, please cite:

```bibtex
@software{cert_sdk,
  title = {CERT: Consistency, Effect, and Reliability Tracking for LLM Systems},
  author = {CERT Contributors},
  year = {2024},
  url = {https://github.com/Javihaus/CERT}
}
```
