# CERT SDK

**Make your multi-agent LLM systems observable and predictable**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

You're building with multiple LLM agents. Your system works... sometimes. You need to answer:

- 🤔 **"Is my agent behaving consistently?"**
- 🔄 **"Do my agents actually coordinate, or just pass messages?"**
- 📊 **"Will my 5-agent pipeline work in production?"**
- ⚠️ **"Why did performance suddenly drop?"**

Traditional monitoring (latency, tokens, errors) doesn't answer these questions.

## What CERT Does

```
┌─────────────────────────────────────────────────────────────┐
│  Your Multi-Agent System                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Agent 1 ──► Agent 2 ──► Agent 3 ──► Final Output          │
│                                                              │
│  CERT measures:                                              │
│  ✓ Consistency: Does Agent 1 behave predictably?           │
│  ✓ Coordination: Do agents improve each other's output?    │
│  ✓ Prediction: Will the pipeline work as expected?         │
│  ✓ Health: Is the system production-ready?                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install cert-sdk
```

### Browse Available Models

```python
import cert

# Show all validated models with baselines
cert.print_models()

# Show only OpenAI models
cert.print_models(provider="openai")

# Get detailed info about a specific model
cert.get_model_info("gpt-4o")

# Compare models side-by-side
from cert.utils.models import compare_models
compare_models("gpt-4o", "grok-3", "gemini-3.5-pro")
```

### Basic Usage - Check Your Agent's Consistency

```python
import asyncio
from cert.models import ModelRegistry
from cert.providers import OpenAIProvider
from cert.providers.base import ProviderConfig

async def check_agent_consistency():
    # 1. Select a validated model
    print("Available models:")
    for model in ModelRegistry.list_models():
        print(f"  - {model.model_id}")

    # 2. Initialize your provider
    config = ProviderConfig(
        api_key="your-api-key",
        model_name="gpt-4o",  # or any validated model
    )
    provider = OpenAIProvider(config)

    # 3. Run the measurement
    from cert.analysis.semantic import SemanticAnalyzer
    from cert.core.metrics import behavioral_consistency

    # Generate 10 responses to the same prompt
    prompt = "Analyze the key factors in project success"
    responses = []
    for i in range(10):
        response = await provider.generate_response(prompt)
        responses.append(response)

    # Calculate consistency
    analyzer = SemanticAnalyzer()
    distances = analyzer.pairwise_distances(responses)
    consistency = behavioral_consistency(distances)

    print(f"Consistency Score: {consistency:.3f}")
    print(f"✓ Good" if consistency > 0.8 else "⚠ Needs attention")

asyncio.run(check_agent_consistency())
```

### Interactive Example

Open the Jupyter notebook:

```bash
jupyter notebook examples/basic_usage.ipynb
```

This will:
1. Show you all validated models with their baselines
2. Let you pick the model you have access to
3. Run consistency and performance measurements
4. Compare your results to known baselines

**Step-by-step walkthrough** with explanations and visualizations.

## Core Concepts

### 1. Consistency - "Does my agent behave predictably?"

```
Same Prompt ──► Agent ──► Response 1: "Focus on timeline and budget"
              │         ──► Response 2: "Consider timeline and costs"
              │         ──► Response 3: "Timeline and budget are key"
              └─ Consistency Score: 0.89 ✓ (Highly consistent)

Same Prompt ──► Agent ──► Response 1: "Focus on timeline"
              │         ──► Response 2: "Team dynamics matter most"
              │         ──► Response 3: "Budget isn't that important"
              └─ Consistency Score: 0.45 ⚠ (Unpredictable)
```

**Why it matters**: Inconsistent agents are unpredictable in production.

### 2. Coordination - "Do my agents work together effectively?"

```
┌──────────────────────────────────────────────────────────┐
│  Independent: Each agent works alone                     │
│  Agent 1 output: Quality = 0.60                         │
│  Agent 2 output: Quality = 0.65                         │
│  Expected (independent): 0.60 × 0.65 = 0.39            │
└──────────────────────────────────────────────────────────┘
                          vs
┌──────────────────────────────────────────────────────────┐
│  Coordinated: Agent 2 sees Agent 1's output             │
│  Pipeline output: Quality = 0.70                        │
│  Coordination Effect: 0.70 / 0.39 = 1.79 ✓             │
│  79% improvement from coordination!                      │
└──────────────────────────────────────────────────────────┘
```

**Why it matters**: Know if your agents actually help each other or just add latency.

### 3. Pipeline Health - "Is my system production-ready?"

```
┌─────────────────────────────────────────────────┐
│  Pipeline Health Score: 0.85 ✓                 │
│                                                  │
│  Based on:                                      │
│  ✓ Prediction accuracy:  95%                   │
│  ✓ Coordination effect:  1.5x improvement      │
│  ✓ Observability:        90% instrumented      │
│                                                  │
│  Status: PRODUCTION READY                       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Pipeline Health Score: 0.42 ⚠                 │
│                                                  │
│  Based on:                                      │
│  ⚠ Prediction accuracy:  45%                   │
│  ⚠ Coordination effect:  0.8x (detrimental!)   │
│  ✓ Observability:        85% instrumented      │
│                                                  │
│  Status: NEEDS INVESTIGATION                    │
└─────────────────────────────────────────────────┘
```

**Why it matters**: Single score tells you if your system is ready to deploy.

## Real-World Use Cases

### Use Case 1: Debugging Agent Inconsistency

**Problem**: Your customer service agent gives different answers to the same question.

```python
# Measure consistency
consistency = measure_agent_consistency(agent)

# Result: 0.65 (inconsistent)
# Action: Adjust temperature, improve prompts, or switch models
```

### Use Case 2: Validating Multi-Agent Pipeline

**Problem**: You added a "reviewer" agent but don't know if it helps.

```python
# Measure coordination effect
gamma = measure_coordination(analyst_agent, reviewer_agent)

# Result: γ = 1.4 (40% improvement)
# Action: Keep the reviewer - it's adding real value
```

### Use Case 3: Pre-Production Validation

**Problem**: Will your 5-agent pipeline work in production?

```python
# Calculate pipeline health
health_score = measure_pipeline_health(your_pipeline)

# Result: H = 0.88
# Action: Safe to deploy with standard monitoring
```

### Use Case 4: Choosing Between Models

**Problem**: Should you use GPT-4o or Gemini for your pipeline?

```python
# Check validated baselines
gpt4o = ModelRegistry.get_model("gpt-4o")
gemini = ModelRegistry.get_model("gemini-3.5-pro")

print(f"GPT-4o:  Consistency={gpt4o.consistency}, Performance={gpt4o.mean_performance}")
print(f"Gemini:  Consistency={gemini.consistency}, Performance={gemini.mean_performance}")

# Result: Gemini has higher consistency (0.895 vs 0.831)
# Action: Use Gemini for predictability-critical applications
```

## Validated Models

The SDK includes pre-measured baselines for these models:

| Model | Provider | Consistency | Performance | Best For |
|-------|----------|-------------|-------------|----------|
| `gemini-3.5-pro` | Google | 0.895 ⭐ | 0.831 ⭐ | Highest stability |
| `grok-3` | xAI | 0.863 | 0.658 | Strong coordination |
| `gpt-4o` | OpenAI | 0.831 | 0.638 | Best prediction accuracy |
| `gpt-4o-mini` | OpenAI | 0.831 | 0.638 | Cost-effective |
| `claude-3-5-haiku-20241022` | Anthropic | 0.831 | 0.595 | Fast and efficient |

### Advanced Features

For models not in the list or domain-specific applications (Healthcare, Legal, Finance):

```bash
jupyter notebook examples/advanced_usage.ipynb
```

This interactive notebook guides you through:
- Measuring baselines for models not in the registry
- Domain-specific applications (Healthcare, Legal, Finance)
- Custom quality scoring with domain keywords
- Registering your custom baselines

## API Overview

### Quick Measurements

```python
from cert.models import ModelRegistry
from cert.core.metrics import (
    behavioral_consistency,
    coordination_effect,
    pipeline_health_score,
)

# Get validated baseline for comparison
baseline = ModelRegistry.get_model("gpt-4o")

# Measure your agent
consistency = behavioral_consistency(your_distances)
print(f"Your agent: {consistency:.3f} vs Baseline: {baseline.consistency:.3f}")

# Measure coordination
gamma = coordination_effect(
    coordinated_performance=0.75,
    independent_performances=[0.60, 0.65]
)
print(f"Coordination effect: {gamma:.2f}x")

# Check pipeline health
health = pipeline_health_score(
    epsilon=0.15,           # prediction error
    gamma_mean=1.35,        # coordination effect
    observability_coverage=0.95  # instrumented fraction
)
print(f"Health score: {health:.2f}")
```

### Working with Providers

```python
from cert.providers import OpenAIProvider, GoogleProvider
from cert.providers.base import ProviderConfig

# Initialize any provider
config = ProviderConfig(
    api_key="your-key",
    model_name="gpt-4o",
)
provider = OpenAIProvider(config)

# Check if model has validated baseline
baseline = provider.get_baseline()
if baseline:
    print(f"✓ Using validated baseline: C={baseline.consistency}")
else:
    print("⚠ Model not validated - measure custom baseline")

# Generate responses (with automatic retry and rate limiting)
response = await provider.generate_response("Your prompt")

# Batch generation for measurements
responses = await provider.batch_generate(
    prompts=["Prompt 1", "Prompt 2"],
    n_samples=10,  # 10 responses per prompt
)
```

## Examples

### Basic Usage (Recommended)
```bash
jupyter notebook examples/basic_usage.ipynb
```
- Interactive model selection
- Automatic baseline comparison
- Measures consistency and performance
- Step-by-step explanations
- Takes 2-3 minutes

### Advanced Features
```bash
jupyter notebook examples/advanced_usage.ipynb
```
- Custom model baselines
- Domain-specific measurements (Healthcare, Legal, Finance)
- Custom quality scoring with keywords
- Baseline registration
- Takes 5-10 minutes

## Architecture Decision Guide

Choose your model based on your needs:

**Need predictability?** → `gemini-3.5-pro` (Highest consistency: 0.895)

**Need strong coordination?** → `grok-3` (Highest γ: 1.625 for 2-agent)

**Need accurate predictions?** → `gpt-4o` (Lowest prediction error: 0.3%)

**Need cost efficiency?** → `gpt-4o-mini` (Same baselines as gpt-4o)

## Common Patterns

### Pattern 1: Validate Before Deploying

```python
# Before production deployment
health = measure_pipeline_health(pipeline)

if health > 0.8:
    deploy_to_production()
elif health > 0.6:
    deploy_with_enhanced_monitoring()
else:
    investigate_and_fix()
```

### Pattern 2: Monitor Consistency Over Time

```python
# Daily consistency check
consistency_today = measure_consistency(agent)
if consistency_today < baseline.consistency - 0.1:
    alert_team("Agent consistency degraded!")
```

### Pattern 3: A/B Test Agent Configurations

```python
# Compare two configurations
config_a_health = measure_health(pipeline_a)
config_b_health = measure_health(pipeline_b)

best_config = config_a if config_a_health > config_b_health else config_b
```

## Development

```bash
git clone https://github.com/Javihaus/CERT.git
cd CERT
pip install -e ".[dev]"

# Run tests
pytest --cov=cert

# Type checking
mypy src/cert

# Formatting
black src/
ruff check src/
```

## Support

- **Documentation**: Full API docs in `/docs`
- **Issues**: https://github.com/Javihaus/CERT/issues
- **Paper**: Based on "CERT: Instrumentation and Metrics for Production LLM Coordination" (Marín, 2025)

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@article{marin2025cert,
  title={CERT: Instrumentation and Metrics for Production LLM Coordination},
  author={Marín, Javier},
  journal={arXiv preprint},
  year={2025}
}
```

---

**Built for engineers shipping multi-agent systems to production.**
