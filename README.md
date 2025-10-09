# CERT SDK

**Production Observability for Multi-Agent LLM Systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Python SDK implementing the CERT (Coordination Error and Risk Tracking) observability framework for multi-agent LLM coordination pipelines.

## Overview

CERT provides systematic observability infrastructure that enables:

- **Behavioral Measurement**: Quantify individual agent consistency and performance
- **Interaction Visibility**: Measure coordination effects in multi-agent pipelines
- **Performance Predictability**: Forecast pipeline behavior and detect degradation

Based on cross-architecture validation across multiple LLM providers, documented in the paper ["CERT: Instrumentation and Metrics for Production LLM Coordination"](https://jmarin.info).

## Key Features

- 🎯 **Five Core Metrics**: Behavioral consistency, coordination effect, prediction error, observability coverage, pipeline health
- 🔄 **Multi-Provider Support**: OpenAI, Anthropic, Google, xAI integrations
- 📊 **Real-time Monitoring**: Streamlit dashboard with historical trends
- 🔬 **Statistical Validation**: Welch's t-test and Cohen's d for significance testing
- 📈 **Production-Ready**: Retry logic, rate limiting, comprehensive error handling
- 🎨 **Quality Scoring**: Multidimensional response evaluation (semantic, coherence, density)

## Quick Start

### Installation

```bash
pip install cert-sdk
```

### Basic Usage

```python
from cert import CERTMonitor
from cert.providers import OpenAIProvider
import numpy as np

# Initialize monitor
monitor = CERTMonitor()

# Add providers
monitor.add_provider(OpenAIProvider(api_key="your-api-key"))

# Measure single agent behavioral consistency
consistency = await monitor.measure_consistency(
    "gpt-4",
    prompt="Analyze the key factors in business strategy",
    n_trials=20
)
print(f"Behavioral Consistency C = {consistency:.3f}")

# Measure two-agent coordination
pipeline = monitor.create_pipeline(["model-1", "model-2"])
metrics = await monitor.evaluate_pipeline(
    pipeline,
    task="Develop a comprehensive digital transformation strategy"
)

print(f"Coordination Effect γ = {metrics.gamma:.3f}")
print(f"Pipeline Health H = {metrics.health_score:.3f}")
```

### Direct Metric Calculation

```python
from cert.core.metrics import (
    behavioral_consistency,
    coordination_effect,
    pipeline_health_score,
)
from cert.analysis.semantic import SemanticAnalyzer

# Measure behavioral consistency from responses
analyzer = SemanticAnalyzer()
responses = [
    "Response 1 to the same prompt...",
    "Response 2 to the same prompt...",
    "Response 3 to the same prompt...",
]

distances = analyzer.pairwise_distances(responses)
consistency = behavioral_consistency(distances)
print(f"C = {consistency:.3f}")

# Calculate coordination effect
gamma = coordination_effect(
    coordinated_performance=0.75,
    independent_performances=[0.60, 0.65]
)
print(f"γ = {gamma:.3f}")  # γ > 1 indicates synergistic coordination

# Compute pipeline health
health = pipeline_health_score(
    epsilon=0.15,  # prediction error
    gamma_mean=1.35,  # coordination effect
    observability_coverage=0.95  # instrumented fraction
)
print(f"H = {health:.2f}")  # H > 0.8 indicates healthy pipeline
```

## Core CERT Metrics

### 1. Behavioral Consistency C(Ai, p)

Measures how consistently an agent responds to identical input using semantic distance variability.

**Formula (Equation 1)**:
$$C(A_i, p) = 1 - \frac{\text{Std}\{d(r_j, r_k)\}}{\text{Mean}\{d(r_j, r_k)\}}$$

**Interpretation**:
- **C → 1**: Stable, predictable behavior
- **C < 0.7**: High variability, requires enhanced monitoring

**Baseline Values from Paper (Table 1)**:

| Architecture | C     |
|--------------|-------|
| Gemini 3.5   | 0.895 |
| Grok 3       | 0.863 |
| Claude 3/3.5 | 0.831 |
| GPT-4        | 0.831 |

### 2. Coordination Effect γ

Measures whether multi-agent coordination produces synergistic or detrimental effects.

**Formula (Equation 3)**:
$$\gamma^P_{i,j}(t) = \frac{\mathbb{E}[P^{\text{coordinated}}_{ij}(t)]}{\mathbb{E}[P^{\text{independent}}_i(t)] \cdot \mathbb{E}[P^{\text{independent}}_j(t)]}$$

**Interpretation**:
- **γ > 1**: Synergistic coordination
- **γ = 1**: No coordination effect
- **γ < 1**: Detrimental interaction

**Two-Agent Baselines from Paper (Table 2)**:

| Architecture | γ     | Statistical Significance |
|--------------|-------|--------------------------|
| Model B      | 1.625 | p < 0.001, d > 0.9       |
| GPT-4        | 1.562 | p < 0.001, d > 0.9       |
| Model A      | 1.462 | p < 0.001, d > 0.9       |
| Gemini 3.5   | 1.137 | p = 0.008, d ≈ 0.66      |

### 3. Pipeline Health Score Hcoord

Composite metric integrating prediction accuracy, coordination strength, and observability coverage.

**Formula (Equation 7)**:
$$H_{\text{coord}}(t) = \frac{1}{1 + \epsilon_{\text{pred}}(t)} \times \min(1, \bar{\gamma}(t)) \times C_{\text{obs}}(t)$$

**Operational Thresholds**:
- **H > 0.8**: Healthy pipeline, standard monitoring
- **0.6 < H ≤ 0.8**: Acceptable, monitor for degradation
- **0.4 < H ≤ 0.6**: Degraded, investigate issues
- **H ≤ 0.4**: Critical, immediate attention required

### 4. Performance Variability Ω

Quantifies behavioral range relative to baseline predictions.

**Operational Interpretation (Table 5)**:

| Architecture | Ω     | Interpretation             |
|--------------|-------|----------------------------|
| Gemini       | 1.729 | Low variability, stable    |
| Model B      | 2.163 | Moderate variability       |
| GPT-4        | 2.320 | Moderate variability       |
| Model A      | 2.852 | High variability, monitor  |

## Architecture Support Matrix

| Provider  | Models            | Baseline Available | Status      |
|-----------|-------------------|--------------------|-------------|
| Provider A | Model Family A   | ✅ (C=0.831)       | ✅ Ready    |
| OpenAI    | GPT-4, GPT-4o     | ✅ (C=0.831)       | ✅ Ready    |
| Provider B | Model Family B   | ✅ (C=0.863)       | ✅ Ready    |
| Google    | Gemini 3.5        | ✅ (C=0.895)       | ✅ Ready    |

## Real-Time Monitoring Dashboard

Launch the Streamlit dashboard for live metrics visualization:

```bash
cert-dashboard --config my_pipeline.yaml
```

Features:
- Live metric values (C, γ, ε, Cobs, Hcoord)
- Historical trends with moving averages
- Cross-architecture comparison charts
- Alert thresholds based on paper's validation
- Export functionality for production logs

## Production Deployment

### Metric Interpretation Thresholds

Based on cross-architecture validation (Table 6):

| Metric               | CV    | Generalization | Interpretation                        |
|----------------------|-------|----------------|---------------------------------------|
| Consistency (C)      | 3.1%  | Excellent      | Stable behavioral measurement         |
| Coordination (γ)     | 13.0% | Moderate       | Architecture-dependent effects        |
| Observability (Cobs) | 0.7%  | Excellent      | Consistent instrumentation            |
| Prediction (ε)       | 42-67%| Poor           | Requires architecture-specific tuning |
| Health (Hcoord)      | 7.8%  | Good           | Actionable aggregated metric          |

### Architecture Selection Criteria (Section 6.4)

**For predictable worst-case performance**:
- Choose: Gemini (Ω=1.729) or GPT-4 (Ω=2.320)
- Accept: Weaker coordination gains (γ=1.137-0.997)

**For strong multi-agent enrichment**:
- Choose: Model A (γ=1.462) or Model B (γ=1.625)
- Accept: Higher prediction uncertainty (ε=19.8-14.0%)

**For balanced deployments**:
- GPT-4: Exceptional prediction accuracy (ε=0.3%) + moderate coordination
- Gemini: High baseline (μ=0.831) + stable execution (Ω=1.729)

## Development

### Setup Development Environment

```bash
git clone https://github.com/Javihaus/CERT.git
cd CERT
pip install -e ".[dev]"
```

### Run Tests

```bash
# Unit tests with coverage
pytest --cov=cert --cov-report=html

# Integration tests
pytest tests/integration/

# Type checking
mypy src/cert

# Linting
ruff check src/
black --check src/
```

## API Documentation

Full API documentation available at [cert-sdk.readthedocs.io](https://cert-sdk.readthedocs.io).

### Key Modules

- **`cert.core.metrics`**: Core CERT metric calculations
- **`cert.providers`**: LLM provider integrations
- **`cert.analysis.semantic`**: Semantic similarity analysis
- **`cert.analysis.quality`**: Response quality scoring (Equation 8)
- **`cert.analysis.statistics`**: Statistical validation utilities
- **`cert.monitoring`**: Real-time dashboard and exporters

## Performance Baselines

### Single-Agent Baseline (Table 1)

| Architecture | C     | μ (mean) | σ (std) |
|--------------|-------|----------|---------|
| Model A      | 0.831 | 0.595    | 0.075   |
| GPT-4        | 0.831 | 0.638    | 0.069   |
| Model B      | 0.863 | 0.658    | 0.062   |
| Gemini 3.5   | 0.895 | 0.831    | 0.090   |

### Five-Agent Pipeline Health (Table 4)

| Rank | Architecture | Hcoord | Primary Driver                              |
|------|--------------|--------|---------------------------------------------|
| 1    | GPT-4        | 0.89   | Exceptional prediction accuracy (ε=0.3%)    |
| 2    | Gemini       | 0.81   | High baseline + moderate prediction         |
| 3    | Model B      | 0.75   | Positive deviation from baseline            |
| 4    | Model A      | 0.73   | Largest positive deviation                  |

## Citation

If you use this SDK in your research, please cite the paper:

```bibtex
@article{marin2025cert,
  title={CERT: Instrumentation and Metrics for Production LLM Coordination},
  author={Marín, Javier},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Documentation**: https://cert-sdk.readthedocs.io
- **Issues**: https://github.com/Javihaus/CERT/issues
- **Paper**: https://jmarin.info/cert

## Acknowledgments

This SDK implements the CERT observability framework as described in "CERT: Instrumentation and Metrics for Production LLM Coordination" (Marín, 2025). Cross-architecture validation was conducted on multiple transformer-based LLM providers.

---

**Built with production reliability in mind** • Not claiming coordination breakthroughs, just making systems more deployable and debuggable.
