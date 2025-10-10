# Runbook: Low Consistency (C < 0.75)

**Alert**: Behavioral consistency score has dropped below acceptable threshold.

---

## Severity

- **Critical**: C < 0.65 - DO NOT deploy
- **Warning**: 0.65 ≤ C < 0.75 - Enhanced monitoring required
- **Acceptable**: C ≥ 0.75 - Standard operations

---

## What This Means

Consistency C measures output variability for identical inputs. Low C indicates:
- High variance in output quality or content
- Unpredictable behavior in production
- Increased risk of user-visible inconsistencies

---

## Immediate Actions (< 5 minutes)

### 1. Check Context Window

**Why**: Context overflow causes erratic behavior.

```python
import cert

# Check current context usage
print(f"Context tokens: {pipeline.count_tokens()}")
print(f"Model limit: {provider.context_limit}")

if pipeline.count_tokens() > provider.context_limit * 0.9:
    print("❌ Context overflow detected!")
```

**Fix**:
- Reduce context length
- Implement summarization between agents
- Switch to larger-context model

### 2. Verify Temperature Setting

**Why**: Temperature > 0 increases sampling variance.

```python
# Check provider configuration
print(f"Temperature: {provider.temperature}")

# For production, should be 0 or very low
if provider.temperature > 0.1:
    print("⚠️  High temperature detected")
    provider.temperature = 0.0
```

**Fix**:
- Set `temperature=0` for deterministic behavior
- If creative output needed, use structured sampling

### 3. Check Recent Changes

**Why**: Prompt or configuration changes affect consistency.

```bash
# Check recent commits to prompts
git log --since="1 week ago" -- prompts/
git log --since="1 week ago" -- config/
```

**Fix**:
- Revert recent prompt changes
- Restore previous configuration
- Recalibrate after changes

---

## Root Cause Analysis (15-30 minutes)

### Diagnostic Steps

```python
import cert
from cert.core.metrics import behavioral_consistency

# 1. Measure current consistency
current_results = await cert.measure_consistency(
    provider=provider,
    n_trials=10
)
print(f"Current C: {current_results['consistency']:.3f}")

# 2. Compare to baseline
baseline = cert.ModelRegistry.get_model(provider.model_name)
print(f"Baseline C: {baseline.consistency:.3f}")
print(f"Drift: {current_results['consistency'] - baseline.consistency:.3f}")

# 3. Inspect sample outputs
print("\nSample outputs for manual inspection:")
for i, output in enumerate(current_results['sample_outputs'][:5]):
    print(f"\n--- Output {i+1} ---")
    print(output[:200])

# 4. Check semantic distances
distances = current_results['semantic_distances']
print(f"\nSemantic distances: mean={distances.mean():.3f}, std={distances.std():.3f}")
```

### Common Root Causes

#### A. Model Version Change

**Symptoms**:
- C dropped suddenly (within days)
- No configuration changes in git
- Affects all prompts equally

**Verification**:
```python
# Check if model behavior changed
baseline_date = cert.ModelRegistry.get_model(provider.model_name).validated_date
print(f"Baseline from: {baseline_date}")
print(f"Current date: {datetime.now()}")

# If > 3 months old, likely model update
```

**Resolution**:
1. Confirm with provider (check OpenAI/Anthropic/Google announcements)
2. Recalibrate baseline:
   ```python
   new_baseline = await cert.calibrate_baseline(
       provider=provider,
       n_trials=20
   )
   ```
3. Update registry
4. Redeploy if new C > 0.75

**Prevention**: Pin specific model versions when available

---

#### B. Prompt Structure Change

**Symptoms**:
- C dropped after specific deployment
- Only affects certain task types
- Git history shows prompt modifications

**Verification**:
```bash
git diff HEAD~1 prompts/
```

**Resolution**:
1. **Option A - Revert**:
   ```bash
   git revert <commit-hash>
   deploy
   ```

2. **Option B - Recalibrate**:
   ```python
   # Measure with new prompts
   new_baseline = await cert.measure_agent(
       provider=provider,
       n_trials=20
   )

   if new_baseline['consistency'] > 0.75:
       print("✅ New prompts acceptable")
       # Update baseline
   else:
       print("❌ New prompts increase variance too much")
       # Revert changes
   ```

**Prevention**:
- A/B test prompt changes
- Measure C before deploying prompt modifications

---

#### C. Context Accumulation

**Symptoms**:
- C starts high, degrades over conversation turns
- Longer conversations → lower consistency
- Single-turn interactions fine

**Verification**:
```python
# Measure C at different context lengths
for context_len in [0, 2000, 5000, 10000]:
    c = measure_with_context_length(provider, context_len)
    print(f"Context {context_len} tokens: C = {c:.3f}")
```

**Resolution**:
1. Implement context summarization:
   ```python
   if context.count_tokens() > 8000:
       context = summarize_context(context)
   ```

2. Use sliding window:
   ```python
   # Keep only last N turns
   context = context[-5:]
   ```

3. Switch to larger-context model:
   ```python
   # GPT-4o: 128K → Claude Sonnet: 200K
   new_provider = cert.create_provider(
       model_name="claude-sonnet-4.5",
       api_key=os.getenv("ANTHROPIC_API_KEY")
   )
   ```

**Prevention**: Monitor context length in production

---

#### D. Domain Shift

**Symptoms**:
- Development C high, production C low
- Affects all new task types
- Sample outputs look reasonable but vary

**Verification**:
```python
# Compare calibration vs production data
print("Calibration prompts:")
print(calibration_prompts[:3])

print("\nProduction prompts:")
print(production_prompts[:3])

# Are they similar in:
# - Complexity?
# - Length?
# - Domain?
```

**Resolution**:
1. Recalibrate on production data:
   ```python
   # Sample real production prompts
   production_sample = sample_production_traffic(n=50)

   # Recalibrate
   production_baseline = await cert.calibrate_baseline(
       provider=provider,
       n_trials=20,
       prompts=production_sample
   )
   ```

2. Create domain-specific baselines:
   ```python
   # Separate baselines per domain
   healthcare_baseline = cert.calibrate_for_domain(
       provider,
       domain="healthcare"
   )
   legal_baseline = cert.calibrate_for_domain(
       provider,
       domain="legal"
   )
   ```

**Prevention**: Always calibrate on production-representative data

---

## Remediation Decision Tree

```
Is C < 0.65?
├─ YES → DO NOT DEPLOY. Fix before production.
└─ NO → Is C < 0.75?
    ├─ YES → Enhanced monitoring. Consider fixes.
    └─ NO → Standard monitoring.

Context > 90% of limit?
├─ YES → Implement summarization/chunking
└─ NO → Continue diagnosis

Temperature > 0?
├─ YES → Set temperature=0
└─ NO → Continue diagnosis

Recent prompt changes?
├─ YES → Revert or recalibrate
└─ NO → Continue diagnosis

C drift > 0.15 from baseline?
├─ YES → Model likely updated. Recalibrate.
└─ NO → Domain mismatch. Use production data.
```

---

## Fix Verification

After implementing fixes:

```python
# 1. Remeasure
new_c = await cert.measure_consistency(provider, n_trials=10)
print(f"New C: {new_c['consistency']:.3f}")

# 2. Verify improvement
if new_c['consistency'] >= 0.75:
    print("✅ Consistency restored")
elif new_c['consistency'] >= 0.70:
    print("⚠️  Improved but still monitoring")
else:
    print("❌ Fix ineffective - escalate")

# 3. Monitor for 24 hours
# Schedule: Check C every 6 hours
# Alert if drops below 0.75 again
```

---

## Escalation

If C remains < 0.75 after:
- Context optimization
- Temperature adjustment
- Prompt reversion
- Recalibration

**Consider**:
1. Model incompatible with task (switch providers)
2. Task inherently high-variance (adjust threshold)
3. Measurement issues (check embedding model)

**Contact**: Platform team / ML ops

---

## Prevention Checklist

- [ ] Pin model versions when possible
- [ ] Monitor context window usage
- [ ] A/B test prompt changes with C measurement
- [ ] Recalibrate quarterly
- [ ] Use production data for baselines
- [ ] Set temperature=0 for deterministic tasks
- [ ] Alert on C drift > 0.10
- [ ] Document baseline dates and conditions

---

## Related Runbooks

- [Low Health Score](./low_health.md)
- [Model Updates](./model_updates.md)
- [Context Window Issues](./context_overflow.md)
