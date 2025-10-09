# CERT Model Registry

## Overview

The CERT Model Registry maintains validated baseline metrics for specific LLM models. **Baselines are model-specific, not provider-generic** - different models from the same provider have different baseline values.

## Why Model-Specific Baselines?

The baseline values (C, μ, σ) from the CERT paper were measured on **specific model versions**:
- ✅ `claude-3-haiku-20240307` (validated)
- ❌ Not "Claude in general"
- ✅ `gpt-4o` (validated)
- ❌ Not "GPT-4 family"
- ✅ `grok-3` (validated)
- ❌ Not "all Grok models"

Different models have different characteristics:
- `claude-3-5-sonnet-20241022` will have different baselines than `claude-3-haiku-20240307`
- `gpt-5-preview` will have different baselines than `gpt-4o`
- Even minor version updates may change behavior

## Validated Models

### Current Registry (4 models from paper validation)

| Model ID | Provider | Family | C | μ | σ | γ(2-agent) |
|----------|----------|--------|---|---|---|------------|
| `claude-3-haiku-20240307` | anthropic | Claude 3 Haiku | 0.831 | 0.595 | 0.075 | 1.462 |
| `gpt-4o` | openai | GPT-4o | 0.831 | 0.638 | 0.069 | 1.562 |
| `grok-3` | xai | Grok 3 | 0.863 | 0.658 | 0.062 | 1.625 |
| `gemini-3.5-pro` | google | Gemini 3.5 Pro | 0.895 | 0.831 | 0.090 | 1.137 |

All values from CERT paper Tables 1-3 (validation date: 2025-01).

## Usage

### 1. List Available Models

```python
from cert.models import ModelRegistry

# List all validated models
models = ModelRegistry.list_models()
for model in models:
    print(model)

# Output:
# Claude 3 Haiku (claude-3-haiku-20240307): C=0.831, μ=0.595, σ=0.075
# GPT-4o (gpt-4o): C=0.831, μ=0.638, σ=0.069
# Grok 3 (grok-3): C=0.863, μ=0.658, σ=0.062
# Gemini 3.5 Pro (gemini-3.5-pro): C=0.895, μ=0.831, σ=0.090

# List by provider
openai_models = ModelRegistry.list_models(provider="openai")

# Get formatted summary
print(ModelRegistry.get_summary())
```

### 2. Check if Model is Validated

```python
from cert.models import ModelRegistry

# Before using a model, check if it has validated baselines
model_id = "gpt-4o"

if ModelRegistry.is_validated(model_id):
    print(f"{model_id} has validated baselines from paper")
    baseline = ModelRegistry.get_model(model_id)
    print(f"C={baseline.consistency:.3f}")
else:
    print(f"{model_id} not validated - need custom baseline measurement")
```

### 3. Get Baseline for Predictions

```python
from cert.models import get_model_baseline
from cert.core.metrics import performance_baseline

# Get validated baseline
baseline = get_model_baseline("gpt-4o")

if baseline:
    # Use for pipeline prediction (Equation 5)
    predicted_performance = performance_baseline(
        agent_means=[baseline.mean_performance] * 3,  # 3-agent pipeline
        gamma_mean=1.4,  # Measured coordination
        alpha=0.93
    )
    print(f"Predicted: {predicted_performance:.3f}")
else:
    print("Model not validated - measure your own baseline")
```

### 4. Compare to Baseline (Reference)

```python
from cert.models import get_model_baseline
from cert.analysis.semantic import SemanticAnalyzer
from cert.core.metrics import behavioral_consistency

# Get paper baseline for comparison
paper_baseline = get_model_baseline("gpt-4o")

# Measure YOUR system's actual consistency
analyzer = SemanticAnalyzer()
your_responses = ["response1", "response2", "response3", ...]
distances = analyzer.pairwise_distances(your_responses)
your_consistency = behavioral_consistency(distances)

# Compare
print(f"Paper baseline: C={paper_baseline.consistency:.3f}")
print(f"Your system:    C={your_consistency:.3f}")

if your_consistency < paper_baseline.consistency - 0.1:
    print("⚠️  Consistency significantly below expected!")
```

## Working with Unlisted Models

### Option 1: Measure Custom Baseline

If your model is not in the registry, measure your own baseline:

```python
from cert.models import ModelRegistry
from cert.analysis.semantic import SemanticAnalyzer
from cert.analysis.quality import QualityScorer
from cert.core.metrics import behavioral_consistency, empirical_performance_distribution
import numpy as np

# Check if model is validated
model_id = "claude-3-5-sonnet-20241022"

if not ModelRegistry.is_validated(model_id):
    print(f"{model_id} not in registry - measuring custom baseline...")

    # Step 1: Measure consistency (20 trials recommended)
    analyzer = SemanticAnalyzer()
    responses = []  # Generate 20 responses to same prompt
    # ... generate responses ...
    distances = analyzer.pairwise_distances(responses)
    consistency = behavioral_consistency(distances)

    # Step 2: Measure performance distribution (15 trials recommended)
    scorer = QualityScorer()
    quality_scores = []
    # ... generate responses and score them ...
    for response in responses:
        score = scorer.score(prompt="...", response=response)
        quality_scores.append(score.composite_score)

    mu, sigma = empirical_performance_distribution(np.array(quality_scores))

    # Step 3: Register custom baseline
    custom_baseline = ModelRegistry.register_custom_baseline(
        model_id=model_id,
        provider="anthropic",
        model_family="Claude 3.5 Sonnet",
        consistency=consistency,
        mean_performance=mu,
        std_performance=sigma,
    )

    print(f"Registered: {custom_baseline}")
```

### Option 2: Use Nearest Validated Model

If measuring a full baseline is impractical, use the nearest validated model as approximation:

```python
from cert.models import ModelRegistry

# You want to use claude-3-5-sonnet but only claude-3-haiku is validated
target_model = "claude-3-5-sonnet-20241022"

if not ModelRegistry.is_validated(target_model):
    # Use claude-3-haiku baseline as approximation
    baseline = ModelRegistry.get_model("claude-3-haiku-20240307")
    print(f"⚠️  Using {baseline.model_id} baseline as approximation")
    print(f"   Results may not be accurate for {target_model}")
```

**Warning**: Using approximate baselines reduces prediction accuracy. Measure custom baselines for production deployments.

## Expanding the Registry

### Contributing New Baselines

To add a new model to the official registry:

1. **Measure baselines** following paper methodology:
   - 20 trials for consistency measurement
   - 15 trials for performance distribution
   - Follow Annex experimental configuration

2. **Validate measurements**:
   - Ensure statistical significance (p < 0.05)
   - Document experimental conditions
   - Use same quality scoring (Equation 8)

3. **Submit contribution**:
   - Open issue with measured values
   - Provide measurement methodology
   - Include model version/date

4. **Registry update**:
   - Add to `src/cert/models.py` `_VALIDATED_MODELS`
   - Update documentation
   - Include in next release

### Baseline Measurement Checklist

- [ ] Model ID and version clearly documented
- [ ] 20+ trials for consistency measurement
- [ ] 15+ trials for performance distribution
- [ ] Same prompt structure as paper (see Annex)
- [ ] Quality scoring uses Equation 8 (semantic 30%, coherence 30%, density 40%)
- [ ] Statistical validation performed
- [ ] Temperature and sampling parameters documented
- [ ] Validation date recorded

## API Reference

### ModelRegistry

**Class Methods:**

```python
# Get specific model baseline
baseline = ModelRegistry.get_model(model_id: str) -> Optional[ModelBaseline]

# List all or filtered models
models = ModelRegistry.list_models(provider: Optional[str] = None) -> List[ModelBaseline]

# List available providers
providers = ModelRegistry.list_providers() -> List[str]

# Check if model is validated
is_valid = ModelRegistry.is_validated(model_id: str) -> bool

# Register custom baseline
baseline = ModelRegistry.register_custom_baseline(
    model_id: str,
    provider: str,
    model_family: str,
    consistency: float,
    mean_performance: float,
    std_performance: float,
    coordination_2agent: Optional[float] = None,
    coordination_5agent: Optional[float] = None,
) -> ModelBaseline

# Get formatted summary
summary = ModelRegistry.get_summary() -> str
```

### ModelBaseline

**Attributes:**

```python
@dataclass(frozen=True)
class ModelBaseline:
    model_id: str                      # e.g., "gpt-4o"
    provider: str                       # e.g., "openai"
    model_family: str                   # e.g., "GPT-4o"
    consistency: float                  # C from Table 1
    mean_performance: float             # μ from Table 1
    std_performance: float              # σ from Table 1
    coordination_2agent: Optional[float]  # γ from Table 2
    coordination_5agent: Optional[float]  # γ from Table 3
    paper_section: str                  # Reference
    validation_date: str                # YYYY-MM
```

## Best Practices

1. **Always check if model is validated** before using baseline values
2. **Measure custom baselines** for production-critical deployments with unlisted models
3. **Document baseline source** (registry vs. custom measurement)
4. **Re-measure baselines** if model version changes
5. **Compare your measurements to registry** to validate implementation
6. **Contribute back** validated baselines for popular models

## Limitations

- Registry contains only 4 models from initial paper validation
- Baselines may change with model updates
- Custom baselines are in-memory only (not persisted)
- Cross-task generalization not guaranteed
- Different tasks may need task-specific baselines

## Future Enhancements

- [ ] Persistent storage for custom baselines
- [ ] Task-specific baseline variants
- [ ] Automatic baseline measurement tool
- [ ] Confidence intervals for baseline values
- [ ] Version tracking for model updates
- [ ] Community-contributed baseline database
