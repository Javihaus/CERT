# Five-Minute Test

**Can you actually use CERT SDK after cloning?**

## The Test

```bash
# 1. Clone
git clone https://github.com/Javihaus/CERT.git
cd CERT

# 2. Install
pip install -e .

# 3. Run something
python quickstart.py
```

## What Works

✅ **Package structure exists**
- `src/cert/` with proper `__init__.py`
- `pyproject.toml` with correct configuration
- Core metrics implementations in `src/cert/core/metrics.py`

✅ **Hardcoded baselines from paper**
- GPT-4o: C=0.831, μ=0.638, σ=0.069
- Claude 3.5 Haiku: C=0.831, μ=0.595, σ=0.075
- Gemini, Grok, etc. with measured values

✅ **Mathematical implementations**
- `behavioral_consistency(distances)` - Equation 1
- `coordination_effect(coordinated, independent)` - Equation 3
- `pipeline_health_score(epsilon, gamma, coverage)` - Equation 7

## What Needs Testing

🔍 **Install and import**
```python
import cert
from cert.models import ModelRegistry
from cert.core.metrics import pipeline_health_score

# Does this work after `pip install -e .`?
models = ModelRegistry.list_models()
print(f"Found {len(models)} validated models")
```

🔍 **Quickstart script**
```bash
python quickstart.py
# Should show baselines and calculate health score WITHOUT API keys
```

🔍 **Basic example**
```bash
python examples/basic_usage.py
# Should work WITH API key, measure actual consistency
```

## Current Status

**Installation**: Package should install with `pip install -e .` after cloning

**Dependencies**: Heavy (sentence-transformers, anthropic, openai, google-generativeai)
- This makes first install slow (~500MB download)
- Consider: Make embeddings optional, use faster alternatives

**Tests**: Empty
- `tests/unit/` - empty
- `tests/integration/` - empty
- `tests/fixtures/` - empty
- Need: `tests/test_smoke.py` that runs without API keys

## What Needs to Exist

### 1. Smoke Test (No API Keys)
```python
# tests/test_smoke.py
def test_import_works():
    import cert

def test_list_models():
    from cert.models import ModelRegistry
    models = ModelRegistry.list_models()
    assert len(models) > 0

def test_calculate_health():
    from cert.core.metrics import pipeline_health_score
    health = pipeline_health_score(0.1, 1.2, 0.9)
    assert 0 < health < 1
```

### 2. Quick Demo (No API Keys)
```python
# quickstart.py - Already created
# Shows baselines, calculates health score with fake data
```

### 3. Real Example (Requires API Key)
```python
# examples/basic_usage.py - Already exists
# Measures actual consistency with 10 API calls
```

## Verdict

**Theory**: ✅ Solid - measured baselines, validated formulas
**Code**: ✅ Exists - implementations match paper
**Packaging**: ✅ Present - pyproject.toml configured
**Tests**: ❌ Missing - no way to verify it works
**Demo**: ⚠️ Requires API keys - needs no-API quickstart

**The Gap**: After cloning, there's no way to verify the package works without:
1. Getting API keys
2. Waiting for large dependency install
3. Running 20+ API calls

**The Fix**: Add `quickstart.py` that uses only hardcoded baselines.
