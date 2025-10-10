# CERT Installation Guide

**Flexible installation options - install only what you need.**

---

## Quick Start

### Option 1: Minimal Core (Metrics Only)

For calculating metrics from pre-collected data:

```bash
pip install cert-sdk
```

**Includes**: numpy, scipy, pydantic, aiohttp, tenacity
**Use for**: Offline metric calculations, testing formulas

---

### Option 2: With Specific Provider

#### OpenAI (GPT-4, GPT-4o, GPT-5)
```bash
pip install cert-sdk[openai,embeddings]
```

#### Anthropic (Claude)
```bash
pip install cert-sdk[anthropic,embeddings]
```

#### Google (Gemini)
```bash
pip install cert-sdk[google,embeddings]
```

#### All Providers
```bash
pip install cert-sdk[all-providers,embeddings]
```

**Note**: `embeddings` includes sentence-transformers for semantic distance calculation (required for full CERT functionality).

---

### Option 3: With Framework Integration

#### LangChain / LangGraph
```bash
pip install cert-sdk[langchain,openai,embeddings]
```

#### CrewAI
```bash
pip install cert-sdk[crewai,openai,embeddings]
```

#### AutoGen
```bash
pip install cert-sdk[autogen,openai,embeddings]
```

#### All Integrations
```bash
pip install cert-sdk[all-integrations,all-providers,embeddings]
```

---

### Option 4: Complete Installation

Everything including visualization dashboard:

```bash
pip install cert-sdk[all]
```

**Includes**:
- All providers (OpenAI, Anthropic, Google)
- All integrations (LangChain, CrewAI, AutoGen)
- Embeddings for semantic analysis
- Streamlit dashboard
- Prometheus monitoring
- Visualization tools (pandas, plotly)

---

## Dependency Groups Reference

| Group | Includes | Use When |
|-------|----------|----------|
| `openai` | openai>=1.0.0 | Using GPT models |
| `anthropic` | anthropic>=0.7.0 | Using Claude models |
| `google` | google-generativeai>=0.3.0 | Using Gemini models |
| `embeddings` | sentence-transformers>=2.2.0 | Need semantic distance (recommended) |
| `langchain` | langchain, langgraph | Using LangChain framework |
| `crewai` | crewai | Using CrewAI framework |
| `autogen` | pyautogen | Using AutoGen framework |
| `dashboard` | streamlit, pandas, plotly | Want visualization UI |
| `prometheus` | prometheus-client | Need metrics export |
| `all-providers` | All provider packages | Using multiple providers |
| `all-integrations` | All framework packages | Using multiple frameworks |
| `all` | Everything | Complete installation |
| `dev` | pytest, black, mypy, etc. | Contributing to CERT |

---

## Installation Examples by Use Case

### Use Case 1: Research / Analysis
**Scenario**: Analyzing multi-agent pipelines with GPT-4

```bash
pip install cert-sdk[openai,embeddings]
```

```python
import cert

provider = cert.create_provider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o"
)

results = await cert.measure_agent(provider, n_consistency_trials=10)
print(f"Consistency: {results['consistency']:.3f}")
```

---

### Use Case 2: Production Deployment with LangChain
**Scenario**: Instrumenting existing LangChain pipeline

```bash
pip install cert-sdk[langchain,openai,embeddings]
```

```python
from cert.integrations.langchain import CERTLangChain
from langgraph.prebuilt import create_react_agent

# Your existing agents
agent1 = create_react_agent(model, tools)
agent2 = create_react_agent(model, tools)

# Add CERT instrumentation
cert_integration = CERTLangChain(
    provider=cert.create_provider(api_key="...", model_name="gpt-4o")
)

pipeline = cert_integration.create_multi_agent_pipeline([
    {"agent": agent1, "agent_id": "agent1", "agent_name": "Researcher"},
    {"agent": agent2, "agent_id": "agent2", "agent_name": "Writer"},
])

result = pipeline({"messages": [input_message]})
cert_integration.print_metrics()
```

---

### Use Case 3: Cost-Conscious Development
**Scenario**: Need to estimate costs before running calibration

```bash
pip install cert-sdk[openai,embeddings]
```

```python
from cert.utils.costs import estimate_calibration_cost

# Estimate first
cost = estimate_calibration_cost("gpt-4o", n_trials=20)
print(cost)

if cost.total_cost_usd > 5.00:
    print("Too expensive, reducing trials")
    n_trials = 10
else:
    n_trials = 20

# Then measure
results = await cert.measure_agent(provider, n_consistency_trials=n_trials)
```

---

### Use Case 4: Multi-Provider Comparison
**Scenario**: Comparing GPT-4, Claude, and Gemini

```bash
pip install cert-sdk[all-providers,embeddings]
```

```python
providers = [
    cert.create_provider(api_key=openai_key, model_name="gpt-4o"),
    cert.create_provider(api_key=anthropic_key, model_name="claude-sonnet-4.5"),
    cert.create_provider(api_key=google_key, model_name="gemini-3.5-pro"),
]

for provider in providers:
    results = await cert.measure_agent(provider)
    print(f"{provider.model_name}: C={results['consistency']:.3f}")
```

---

### Use Case 5: Offline Testing (No API Keys)
**Scenario**: Learning CERT metrics without API costs

```bash
pip install cert-sdk
```

```python
# Use hardcoded baselines from paper
import cert

baseline = cert.ModelRegistry.get_model("gpt-4o")
print(f"GPT-4o baseline: C={baseline.consistency}, μ={baseline.mean_performance}")

# Test metric calculations
from cert.core.metrics import behavioral_consistency
import numpy as np

distances = np.array([0.12, 0.15, 0.13, 0.14, 0.16])
c = behavioral_consistency(distances)
print(f"Calculated C: {c:.3f}")
```

---

## Verification

After installation, verify with:

```bash
python -c "import cert; print(cert.__version__)"
```

### Check Available Providers
```python
import cert
print("Available providers:")
try:
    import openai
    print("✓ OpenAI")
except ImportError:
    print("✗ OpenAI (install with: pip install cert-sdk[openai])")

try:
    import anthropic
    print("✓ Anthropic")
except ImportError:
    print("✗ Anthropic (install with: pip install cert-sdk[anthropic])")

try:
    import google.generativeai
    print("✓ Google")
except ImportError:
    print("✗ Google (install with: pip install cert-sdk[google])")
```

### Check Available Integrations
```python
print("Available integrations:")
try:
    import langchain
    print("✓ LangChain")
except ImportError:
    print("✗ LangChain (install with: pip install cert-sdk[langchain])")

try:
    import crewai
    print("✓ CrewAI")
except ImportError:
    print("✗ CrewAI (install with: pip install cert-sdk[crewai])")

try:
    import autogen
    print("✓ AutoGen")
except ImportError:
    print("✗ AutoGen (install with: pip install cert-sdk[autogen])")
```

---

## Upgrading

### From Earlier Versions
```bash
pip install --upgrade cert-sdk
```

### With Specific Groups
```bash
pip install --upgrade cert-sdk[openai,embeddings]
```

### Force Reinstall
```bash
pip install --force-reinstall cert-sdk[all]
```

---

## Uninstalling

```bash
pip uninstall cert-sdk
```

**Note**: This only removes CERT. Optional dependencies (openai, langchain, etc.) remain installed. Remove them separately if needed:

```bash
pip uninstall openai anthropic langchain crewai pyautogen
```

---

## Troubleshooting

### ImportError: No module named 'sentence_transformers'
```bash
pip install cert-sdk[embeddings]
```

### ImportError: No module named 'langchain'
```bash
pip install cert-sdk[langchain]
```

### API Key Errors
Ensure environment variables are set:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

### Version Conflicts
Create a fresh virtual environment:
```bash
python -m venv cert-env
source cert-env/bin/activate  # On Windows: cert-env\Scripts\activate
pip install cert-sdk[all]
```

---

## Development Installation

For contributing to CERT:

```bash
git clone https://github.com/Javihaus/CERT.git
cd CERT
pip install -e ".[dev,all]"
```

Run tests:
```bash
pytest tests/ -v
```

---

## Next Steps

After installation:
1. **Try the quickstart**: `python quickstart.py`
2. **Estimate costs**: See `src/cert/utils/costs.py` usage
3. **Read failure modes**: `docs/failure_modes.md`
4. **Check operational runbooks**: `docs/runbooks/`

---

**Questions?** See [README.md](README.md) or [open an issue](https://github.com/Javihaus/CERT/issues).
