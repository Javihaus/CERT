# CERT Failure Modes and Limitations

**Document what actually breaks and when CERT metrics become invalid.**

---

## Critical Failure Modes

### 1. Context Window Overflow

**Symptom**: Semantic distance calculations fail or return nonsensical values (C < 0.5 unexpectedly).

**Root Cause**:
- When accumulated context in sequential pipelines exceeds model's context window (e.g., 128K tokens for GPT-4o)
- Model begins truncating or dropping context
- Outputs become inconsistent due to lost information, not natural variance

**Detection**:
```python
if total_context_tokens > model_context_limit * 0.9:
    logger.warning("Approaching context limit - C metric may be invalid")
```

**Mitigation**:
- Monitor context window usage explicitly
- Implement context summarization between agents
- Use models with larger context windows
- Recalibrate baseline with actual production context lengths

**When to Invalidate Metrics**:
- If context exceeds 90% of model limit: C and γ become unreliable
- Outputs reflect truncation behavior, not coordination

---

### 2. Prompt Structure Changes Mid-Deployment

**Symptom**: C metric drops >0.15 from baseline without model version change.

**Root Cause**:
- Changing prompt templates alters the semantic space of outputs
- Baseline consistency was measured on old prompt structure
- New prompts may have different natural variance characteristics

**Example**:
```python
# Before: High structure → C = 0.85
prompt_old = "Answer in exactly 3 bullet points: {question}"

# After: Free form → C = 0.70 (not a bug, different task)
prompt_new = "Discuss the following question: {question}"
```

**Detection**:
- Log prompt template versions alongside metrics
- Alert when C drops >0.10 between consecutive measurements
- Check git history for prompt changes

**Mitigation**:
- Version prompts explicitly
- Recalibrate baseline after ANY prompt changes
- Treat prompt changes like model version updates

**When to Invalidate Metrics**:
- After any prompt template modification
- When output format requirements change
- If instruction style shifts (imperative → conversational)

---

### 3. Model Version Updates

**Symptom**: Baseline metrics no longer predict observed performance (ε > 0.20).

**Root Cause**:
- OpenAI/Anthropic/Google update models behind stable identifiers
- `gpt-4o` from October 2025 ≠ `gpt-4o` from January 2026
- Behavior characteristics shift without notification

**Detection**:
```python
current_c = measure_consistency(provider)
baseline_c = ModelRegistry.get_model("gpt-4o").consistency

if abs(current_c - baseline_c) > 0.15:
    logger.error("Baseline drift detected - model may have updated")
```

**Mitigation**:
- Pin model versions when possible (e.g., `gpt-4o-2025-10-10`)
- Run weekly baseline drift checks
- Maintain model snapshot dates in registry
- Recalibrate quarterly regardless of drift detection

**When to Invalidate Metrics**:
- When model provider announces updates
- If C drift > 0.15 from registry baseline
- If ε suddenly increases > 0.15

---

### 4. Domain Shift

**Symptom**: Pipeline works in development but H < 0.60 in production.

**Root Cause**:
- Baselines calibrated on synthetic/test data
- Production data has different complexity/ambiguity
- Model behavior differs across domains (works for summaries, fails for code)

**Example**:
```python
# Calibrated on clean academic text
baseline_data = ["Summarize this research paper...", ...]
calibrate_baseline(provider, prompts=baseline_data)  # C = 0.85

# Production: messy user queries
production_data = ["my code doesnt work lol help ???", ...]
# Actual C in production: 0.65 (not a bug - different domain)
```

**Detection**:
- Compare baseline measurement domain vs production domain
- Monitor H score across different task types
- Sample production traffic for re-calibration

**Mitigation**:
- **ALWAYS** calibrate on production-representative data
- Create domain-specific baselines (healthcare, legal, code, etc.)
- Use custom keywords and evaluation criteria per domain

**When to Invalidate Metrics**:
- If calibration data doesn't match production distribution
- When deploying to new domain (healthcare → legal)
- If task complexity increases (summaries → reasoning)

---

### 5. Token Limit Rate Limiting / Throttling

**Symptom**: Timeouts, retries, degraded performance during measurement.

**Root Cause**:
- n=20 trials exceed provider rate limits
- Parallel requests get throttled
- Measurements take orders of magnitude longer than expected

**Example**:
```python
# Naive calibration
for i in range(20):
    result = await provider.generate(prompt)  # May hit rate limit
```

**Detection**:
- Monitor API error codes (429, 503)
- Track measurement duration (calibration taking >10 minutes)
- Log retry attempts

**Mitigation**:
```python
from cert.utils.costs import estimate_calibration_cost

# Check cost AND rate limits
estimate = estimate_calibration_cost("gpt-4o", n_trials=20)
print(f"This will make {estimate.n_api_calls} API calls")

# Implement backoff
import tenacity

@tenacity.retry(
    wait=tenacity.wait_exponential(min=1, max=60),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(RateLimitError)
)
async def measure_with_retry():
    return await cert.measure_agent(provider)
```

**When to Invalidate Metrics**:
- If >20% of measurement calls failed/retried
- If measurements took >5x expected duration
- Remeasure with proper rate limit handling

---

### 6. Embedding Model Misalignment

**Symptom**: C values don't correlate with observed output quality.

**Root Cause**:
- Semantic distance calculated using wrong embedding model
- Embedding model doesn't capture domain-specific semantics
- Using generic embeddings for specialized domains (medical, legal)

**Example**:
```python
# Generic embeddings fail for code
output1 = "def foo(): return x + 1"
output2 = "def foo(): return 1 + x"
# High semantic similarity but DIFFERENT behavior if x is string!
```

**Detection**:
- Manual inspection of "inconsistent" outputs
- If C is high but human evaluation shows variance
- Domain experts disagree with consistency scores

**Mitigation**:
- Use domain-specific embedding models
- Implement custom semantic distance functions
- Combine embeddings with rule-based checks for critical domains

**When to Invalidate Metrics**:
- When domain requires specialized semantics (code, math, legal)
- If embedding model language ≠ output language
- For outputs with critical semantic nuances

---

### 7. Non-Deterministic vs Deterministic Behavior

**Symptom**: C measurements wildly inconsistent across runs.

**Root Cause**:
- Temperature=0 still produces variance due to sampling
- Some models (GPT-4) more deterministic than others (Claude)
- Baseline assumes model's natural variance, not measurement error

**Example**:
```python
# Measuring C with temperature > 0
provider = cert.create_provider(
    api_key="...",
    model_name="gpt-4o",
    temperature=0.7  # High variance expected!
)
result = measure_consistency(provider)  # C = 0.60 (not a bug)
```

**Detection**:
- Check provider configuration temperature setting
- Compare C across multiple measurement runs (should be stable)
- Verify model name matches baseline registry

**Mitigation**:
- **ALWAYS** use temperature=0 for baseline calibration
- Document temperature setting with measurements
- Create separate baselines per temperature if needed

**When to Invalidate Metrics**:
- If temperature ≠ 0 during calibration
- If model configuration changed (temperature, top_p, etc.)
- Recalibrate with production-matching parameters

---

## Operational Thresholds

### When Metrics Are Valid

✅ **Use CERT metrics when**:
- Prompts haven't changed since calibration
- Model version is stable and known
- Domain matches calibration data
- Context windows within safe limits (< 90% of max)
- Measurement settings match production config

### When Metrics Are Invalid

❌ **DO NOT trust CERT metrics when**:
- Context exceeds 90% of model limit
- Prompts changed since last calibration
- Model version unknown or recently updated
- Domain shifted from calibration data
- Temperature or sampling params changed
- Measurement failures / rate limit issues occurred

---

## Recalibration Triggers

Recalibrate baselines when:

1. **Scheduled**: Every 3 months minimum
2. **Model Updates**: Any known model version change
3. **Drift Detection**: C drift > 0.15 from registry
4. **Prompt Changes**: Any template modifications
5. **Domain Shift**: New task types or data distributions
6. **Config Changes**: Temperature, max_tokens, or sampling params

---

## Debug Checklist

When metrics seem wrong:

```
[ ] Check context window usage
[ ] Verify prompt template versions
[ ] Compare model version to baseline
[ ] Confirm domain match with calibration
[ ] Review temperature and sampling settings
[ ] Check for rate limiting / API errors
[ ] Validate embedding model for domain
[ ] Inspect sample outputs manually
[ ] Compare to registry baseline
[ ] Check calibration data representativeness
```

---

## Recovery Procedures

### Scenario: Unexpected Low C Score

```python
# 1. Verify measurement validity
print(f"Context tokens: {context_tokens}")
print(f"Temperature: {provider.temperature}")
print(f"Model version: {provider.model_name}")

# 2. Compare to baseline
baseline = cert.ModelRegistry.get_model(provider.model_name)
print(f"Expected C: {baseline.consistency}")
print(f"Measured C: {current_c}")

# 3. If drift > 0.15, recalibrate
if abs(current_c - baseline.consistency) > 0.15:
    logger.warning("Significant drift - recalibrating")
    new_baseline = await cert.calibrate_baseline(
        provider=provider,
        n_trials=20,
        domain_prompts=production_prompts
    )
    # Update registry with new baseline
```

### Scenario: High Prediction Error (ε > 0.20)

```python
# 1. Check for model updates
print("Check provider announcements for model updates")

# 2. Verify context window not exceeded
if context_tokens > model_limit * 0.9:
    logger.error("Context window issue - split pipeline")

# 3. Recalculate baseline with current behavior
new_baseline = await cert.measure_agent(provider, n_trials=20)

# 4. Update prediction model
baseline_registry.update(model_name, new_baseline)
```

---

## Known Limitations (By Design)

These are NOT bugs - CERT cannot and should not handle:

1. **Semantic correctness**: CERT measures variance, not accuracy
2. **Hallucination detection**: Use specialized tools (factuality checkers)
3. **Reasoning capabilities**: CERT is behavior characterization, not intelligence test
4. **Novel task performance**: Baselines only valid for similar task distributions
5. **Security issues**: Jailbreaks, prompt injection require separate tooling
6. **Cross-language**: Baselines are language-specific (English validation only)
7. **Multimodal**: Image/audio inputs not supported in baseline registry

---

For operational guidance when metrics indicate issues, see:
- `/docs/runbooks/low_consistency.md`
- `/docs/runbooks/low_gamma.md`
- `/docs/runbooks/low_health.md`
