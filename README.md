# CERT: Statistical Observability for Multi-Agent LLM Systems

[![PyPI](https://img.shields.io/pypi/v/cert-sdk)](https://pypi.org/project/cert-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Production monitoring for nondeterministic LLM pipelines.**

Your multi-agent system works... sometimes. Same input produces different quality outputs. You can't answer basic questions: Why did performance drop last week? Will adding a third agent help? Should you deploy this?

CERT provides three metrics that make sequential LLM pipelines measurable:
- **Consistency (C)**: Output quality variance for identical inputs
- **Context Effect (γ, γ_norm)**: Performance change from sequential processing  
- **Health Score (H)**: Go/no-go deployment metric

**Validated across 8 models (GPT-5, Claude Sonnet 4.5, Gemini 3.5, Grok 3, GPT-4o, and more) with 300+ measurements.**

---

## Start in 30 Seconds

Run this without API keys—uses hardcoded baselines from validation:

```bash
git clone https://github.com/Javihaus/CERT.git && cd CERT
pip install -e .
python quickstart.py
```

Output:

```
✓ Claude Sonnet 4.5: C=0.892, μ=0.745, γ_norm=1.116 → H=0.78
✓ GPT-5:            C=0.702, μ=0.543, γ_norm=1.382 → H=0.71
✓ Gemini 3.5 Pro:   C=0.895, μ=0.831, γ_norm=1.066 → H=0.84
✓ GPT-4o:           C=0.831, μ=0.638, γ_norm=1.250 → H=0.76
```

**What this tells you:** Framework measures real variance. Gemini = highest individual quality, lowest context effect. GPT-5 = "team player" optimized for multi-agent pipelines.

See: `quickstart.py`, `tests/test_smoke.py`

---

## Quick Start

### Install

```bash
pip install cert-sdk  # When published
```

**OR**

```bash
pip install -e .  # From source
```

### Measure Single Agent

```python
import cert

provider = cert.create_provider(api_key="sk-...", model_name="gpt-4o")

# ~2 minutes for n=10 trials
results = await cert.measure_agent(provider, n_consistency_trials=10)

print(f"Consistency: {results['consistency']:.3f}")
print(f"Performance: μ={results['mean_performance']:.3f}")

# Compare to validated baseline
baseline = cert.ModelRegistry.get_model("gpt-4o")
if results['consistency'] < baseline.consistency - 0.10:
    print("⚠️  Degraded vs baseline")
```

### Measure Pipeline

```python
# 2-agent pipeline
pipeline = await cert.measure_pipeline(
    agents=[provider1, provider2],
    n_trials=15
)

print(f"Context Effect: γ_norm={pipeline['gamma_norm']:.2f}")
print(f"Health Score: H={pipeline['health_score']:.2f}")

# Deployment decision
if pipeline['health_score'] > 0.80:
    deploy_to_production()
elif pipeline['health_score'] > 0.60:
    deploy_with_monitoring()
else:
    investigate_issues()
```

---

## Model Baselines

First measured baselines for GPT-5 and Claude Sonnet 4.5. Validated across 8 models, 4 providers.

| Model | C | μ | γ_norm | Profile |
|-------|---|---|--------|---------|
| claude-sonnet-4.5 🆕 | 0.892 | 0.745 | 1.116 | Highest consistency, balanced |
| gemini-3.5-pro | 0.895 | 0.831 | 1.066 | Individual specialist |
| gpt-4o | 0.831 | 0.638 | 1.250 | Production workhorse |
| grok-3 | 0.863 | 0.658 | 1.275 | Strong context propagation |
| gpt-5 🆕 | 0.702 | 0.543 | 1.382 | "Team player" - optimized for multi-agent |
| claude-3-5-haiku | 0.831 | 0.595 | 1.209 | Cost-effective |
| gpt-4o-mini | 0.831 | 0.638 | 1.250 | Fastest |

**Selection Guide:**
- **High-reliability**: Claude Sonnet 4.5, Gemini (C > 0.89)
- **Multi-agent pipelines**: GPT-5, Grok 3 (γ_norm > 1.27)
- **Cost-optimized**: GPT-4o-mini, Claude Haiku

```python
# Get baseline
baseline = cert.ModelRegistry.get_model("gpt-4o")

# List models
cert.print_models()
cert.print_models(provider="anthropic")
```

---

## Integration

### LangChain

```python
from cert.integrations.langchain import CERTLangChain

# Existing agents
agent1 = create_react_agent(model, tools)
agent2 = create_react_agent(model, tools)

# Add instrumentation
cert_integration = CERTLangChain(
    provider=cert.create_provider(api_key="...", model_name="gpt-4o")
)

pipeline = cert_integration.create_multi_agent_pipeline([
    {"agent": agent1, "agent_id": "researcher"},
    {"agent": agent2, "agent_id": "writer"},
])

result = pipeline({"messages": [input]})
cert_integration.print_metrics()
```

### CrewAI

```python
from cert.integrations.crewai import CERTCrewAI

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])

cert_integration = CERTCrewAI(
    provider=cert.create_provider(api_key="...", model_name="gpt-4o")
)

instrumented_crew = cert_integration.wrap_crew(crew)
result = instrumented_crew.kickoff()
cert_integration.print_metrics()
```

### AutoGen

```python
from cert.integrations.autogen import CERTAutoGen

agents = [researcher, writer, critic]

cert_integration = CERTAutoGen(
    provider=cert.create_provider(api_key="...", model_name="gpt-4o")
)

groupchat = cert_integration.create_instrumented_groupchat(agents, max_round=10)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)
user_proxy.initiate_chat(manager, message="Task")
cert_integration.print_metrics()
```

---

## What It Measures

### Consistency (C)

Output quality variance for identical inputs.

**Interpretation:**
- C > 0.85: Production ready (low variance)
- C 0.75-0.85: Monitor closely
- C < 0.75: Investigate (high variance)

**Use:** Detect model drift, validate prompt changes, assess reliability.

### Context Effect (γ, γ_norm)

Performance change from sequential context accumulation.

**Formulas:**
- γ = P_sequential / (μ₁ × μ₂ × ... × μₙ × α^(n-1))
- γ_norm = γ^(1/n) (normalized for cross-length comparison)

**Interpretation:**
- γ_norm > 1.5: Strong context benefit
- γ_norm 1.2-1.5: Moderate propagation
- γ_norm 1.0-1.2: Weak context effect
- γ_norm < 1.0: Context degradation

**Use:** Decide if adding agents helps, compare 2-agent vs 5-agent architectures.

### Health Score (H)

Composite operational metric for deployment decisions.

**Formula:** H = (1/(1+ε)) × min(1, γ_norm) × C_obs

**Interpretation:**
- H > 0.80: Deploy to production
- H 0.60-0.80: Deploy with monitoring
- H < 0.60: Do not deploy

**Use:** Go/no-go decisions, architecture validation, continuous monitoring.

---

## Production Workflows

### Detect Model Drift

```python
# Baseline
baseline = measure_agent(provider)  # C=0.83

# Week later
current = measure_agent(provider)   # C=0.71

if current['consistency'] < baseline['consistency'] - 0.10:
    alert("⚠️  Consistency degraded 14%")
```

### Architecture Selection

```python
# Test configurations
pipeline_a = measure_pipeline([researcher, writer])           # H=0.72
pipeline_b = measure_pipeline([researcher, writer, reviewer]) # H=0.81

# Decision: H increased 12% → keep reviewer
```

### Pre-Deployment Validation

```python
health = measure_pipeline_health(new_pipeline)

if health > 0.80:
    deploy_to_production()
elif health > 0.60:
    deploy_with_monitoring(alert_threshold=0.70)
else:
    run_diagnostics()
```

---

## Scope

### Designed for:
- Sequential LLM pipelines (Research→Writer→Editor)
- Production debugging (why did quality drop?)
- Architecture comparison (which topology?)
- Drift detection (has behavior changed?)

### Not designed for:
- Star/mesh topologies (requires different baselines)
- Real-time quality assessment (needs 10-20 trials)
- Hallucination detection (orthogonal problem)

### Measures:
- Statistical variance in transformer outputs
- Performance changes from context accumulation
- Attention mechanism behavior with extended context

### Does not measure:
- Intelligence, reasoning, or understanding
- Agent coordination or collaboration
- Root cause explanations (detects problems, doesn't diagnose)

---

## Documentation

- **Operational Guide** - Deployment thresholds, monitoring strategies
- **API Reference** - Complete function documentation
- **Methodology** - Mathematical framework, validation approach
- **Examples** - Jupyter notebooks for common workflows

---

## Examples

### Basic Usage

```bash
jupyter notebook examples/basic_usage.ipynb
```

Measure individual agent, interpret metrics (2-3 min)

### LangChain Pipeline

```bash
jupyter notebook examples/langchain_research_writer_pipeline.ipynb
```

3-agent sequential pipeline with health score (5-10 min)

### Advanced

```bash
jupyter notebook examples/advanced_usage.ipynb
```

Domain-specific baselines, custom quality scoring (10-15 min)

---

## Citation

```bibtex
@article{marin2025cert,
  title={CERT: Instrumentation and Metrics for Production LLM Sequential Processing},
  author={Marín, Javier},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```
