#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CERT Baseline Measurement for Claude Sonnet 4.5

This script measures the baseline metrics for Claude Sonnet 4.5:
- Behavioral Consistency (C): 20 trials on identical prompt
- Performance Baseline (mu, sigma): 15 prompts x 5 samples each
- Context Propagation Effect (gamma): 5 prompt pairs

Replicates exact methodology from CERT paper.
"""

import anthropic
import asyncio
import statistics
import time
from typing import List, Tuple
import os

# Model configuration
MODEL_NAME = "claude-sonnet-4-20250514"  # Claude Sonnet 4.5
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Get API key from environment
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    exit(1)

client = anthropic.AsyncAnthropic(api_key=api_key)


async def generate_response(prompt: str, temperature: float = TEMPERATURE) -> str:
    """Generate a single response from Claude."""
    try:
        message = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


def score_response(response: str, task: str) -> float:
    """
    Score response quality on 0-1 scale.

    Scoring criteria:
    - Relevance to task (0.3)
    - Completeness (0.3)
    - Clarity and structure (0.2)
    - Accuracy/validity (0.2)
    """
    score = 0.0

    # Basic checks
    if not response or len(response) < 50:
        return 0.1

    response_lower = response.lower()

    # Relevance (0.3): Check if response addresses key terms
    relevance = 0.0
    task_keywords = set(task.lower().split())
    task_keywords = {w for w in task_keywords if len(w) > 3}

    if task_keywords:
        matched = sum(1 for kw in task_keywords if kw in response_lower)
        relevance = min(0.3, (matched / len(task_keywords)) * 0.3)

    score += relevance

    # Completeness (0.3): Length and structure
    if len(response) > 200:
        score += 0.1
    if len(response) > 500:
        score += 0.1
    if any(marker in response for marker in ['\n\n', '1.', '2.', '-', '*']):
        score += 0.1

    # Clarity (0.2): Sentence structure
    sentences = response.count('.') + response.count('!') + response.count('?')
    if sentences >= 3:
        score += 0.1
    if sentences >= 5:
        score += 0.1

    # Accuracy (0.2): No obvious errors or refusals
    if not any(phrase in response_lower for phrase in ['i cannot', 'i\'m unable', 'i can\'t']):
        score += 0.1
    if len(response) > 100 and not response.startswith("I apologize"):
        score += 0.1

    return min(1.0, score)


# Test prompts for behavioral consistency (20 trials, 1 prompt)
CONSISTENCY_PROMPT = "Explain the key factors that contribute to successful business strategy in competitive markets."

# Test prompts for performance baseline (15 prompts × 5 samples)
BASELINE_PROMPTS = [
    "Analyze the impact of artificial intelligence on modern healthcare systems.",
    "Describe the main challenges facing renewable energy adoption globally.",
    "Explain how blockchain technology is transforming financial services.",
    "Discuss the psychological factors that influence consumer purchasing decisions.",
    "Outline the key principles of effective team leadership in organizations.",
    "Analyze the role of data privacy in the digital age.",
    "Explain the economic implications of remote work trends.",
    "Describe the environmental impact of fast fashion industry.",
    "Discuss the future of autonomous vehicles in urban transportation.",
    "Analyze the challenges of scaling startup companies.",
    "Explain the importance of cybersecurity in critical infrastructure.",
    "Describe the evolution of social media's influence on society.",
    "Discuss the role of education in addressing income inequality.",
    "Analyze the impact of climate change on global food security.",
    "Explain the principles of sustainable urban development.",
]

# Prompt pairs for context propagation (5 pairs)
PROPAGATION_PAIRS = [
    (
        "Analyze the current trends in renewable energy adoption.",
        "Based on these trends, what are the main barriers to widespread implementation?"
    ),
    (
        "Explain the concept of machine learning in simple terms.",
        "How would you apply these concepts to solve real-world business problems?"
    ),
    (
        "Describe the key challenges in modern healthcare systems.",
        "What innovative solutions could address these challenges?"
    ),
    (
        "Outline the principles of effective digital marketing.",
        "How do these principles change for B2B versus B2C markets?"
    ),
    (
        "Analyze the impact of globalization on local economies.",
        "What policies could help communities adapt to these changes?"
    ),
]


async def measure_consistency(n_trials: int = 20) -> Tuple[float, List[float]]:
    """
    Measure behavioral consistency C.

    Returns: (consistency_score, list_of_scores)
    """
    print(f"\n{'='*70}")
    print("MEASURING BEHAVIORAL CONSISTENCY (C)")
    print(f"{'='*70}")
    print(f"Prompt: {CONSISTENCY_PROMPT[:60]}...")
    print(f"Trials: {n_trials}")
    print()

    scores = []

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}...", end=" ", flush=True)
        response = await generate_response(CONSISTENCY_PROMPT)
        score = score_response(response, CONSISTENCY_PROMPT)
        scores.append(score)
        print(f"score={score:.3f}")

        # Rate limiting
        await asyncio.sleep(1)

    # Calculate consistency (1 - coefficient of variation)
    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0
    cv = std_score / mean_score if mean_score > 0 else 0
    consistency = max(0, 1 - cv)

    print(f"\nResults:")
    print(f"  Mean score: {mean_score:.3f}")
    print(f"  Std dev: {std_score:.3f}")
    print(f"  Coefficient of variation: {cv:.3f}")
    print(f"  Behavioral Consistency C: {consistency:.3f}")

    return consistency, scores


async def measure_performance_baseline(n_prompts: int = 15, n_samples: int = 5) -> Tuple[float, float, List[float]]:
    """
    Measure performance baseline (mu, sigma).

    Returns: (mean_performance, std_performance, all_scores)
    """
    print(f"\n{'='*70}")
    print("MEASURING PERFORMANCE BASELINE (mu, sigma)")
    print(f"{'='*70}")
    print(f"Prompts: {n_prompts}")
    print(f"Samples per prompt: {n_samples}")
    print(f"Total samples: {n_prompts * n_samples}")
    print()

    all_scores = []

    for i, prompt in enumerate(BASELINE_PROMPTS[:n_prompts], 1):
        print(f"\nPrompt {i}/{n_prompts}: {prompt[:50]}...")
        prompt_scores = []

        for j in range(n_samples):
            print(f"  Sample {j+1}/{n_samples}...", end=" ", flush=True)
            response = await generate_response(prompt)
            score = score_response(response, prompt)
            prompt_scores.append(score)
            all_scores.append(score)
            print(f"score={score:.3f}")

            # Rate limiting
            await asyncio.sleep(1)

        print(f"  Prompt mean: {statistics.mean(prompt_scores):.3f}")

    mean_perf = statistics.mean(all_scores)
    std_perf = statistics.stdev(all_scores) if len(all_scores) > 1 else 0

    print(f"\nResults:")
    print(f"  Mean Performance (mu): {mean_perf:.3f}")
    print(f"  Std Performance (sigma): {std_perf:.3f}")

    return mean_perf, std_perf, all_scores


async def measure_context_propagation(n_pairs: int = 5) -> Tuple[float, List[float]]:
    """
    Measure context propagation effect gamma.

    Returns: (gamma, list_of_ratios)
    """
    print(f"\n{'='*70}")
    print("MEASURING CONTEXT PROPAGATION EFFECT (gamma)")
    print(f"{'='*70}")
    print(f"Prompt pairs: {n_pairs}")
    print()

    ratios = []

    for i, (prompt1, prompt2) in enumerate(PROPAGATION_PAIRS[:n_pairs], 1):
        print(f"\nPair {i}/{n_pairs}")
        print(f"  Prompt 1: {prompt1[:50]}...")
        print(f"  Prompt 2: {prompt2[:50]}...")

        # Baseline: Prompt 2 alone
        print(f"  Generating baseline (P2 alone)...", end=" ", flush=True)
        baseline_response = await generate_response(prompt2)
        baseline_score = score_response(baseline_response, prompt2)
        print(f"score={baseline_score:.3f}")

        await asyncio.sleep(1)

        # Sequential: Prompt 1 then Prompt 2
        print(f"  Generating sequential (P1→P2)...", end=" ", flush=True)

        # In sequential mode, we concatenate with context marker
        sequential_prompt = f"{prompt1}\n\n{prompt2}"
        sequential_response = await generate_response(sequential_prompt)
        sequential_score = score_response(sequential_response, prompt2)
        print(f"score={sequential_score:.3f}")

        await asyncio.sleep(1)

        # Calculate ratio
        if baseline_score > 0:
            ratio = sequential_score / baseline_score
        else:
            ratio = 1.0

        ratios.append(ratio)
        print(f"  Ratio (sequential/baseline): {ratio:.3f}")

    gamma = statistics.mean(ratios)

    print(f"\nResults:")
    print(f"  Mean ratio: {gamma:.3f}")
    print(f"  Context Propagation Effect (gamma): {gamma:.3f}")

    return gamma, ratios


async def main():
    """Run complete baseline measurement."""
    print("="*70)
    print("CERT BASELINE MEASUREMENT")
    print("Model: Claude Sonnet 4.5 (claude-sonnet-4-20250514)")
    print("="*70)

    start_time = time.time()

    # Measure all three metrics
    consistency, _ = await measure_consistency(n_trials=20)
    mean_perf, std_perf, _ = await measure_performance_baseline(n_prompts=15, n_samples=5)
    gamma, _ = await measure_context_propagation(n_pairs=5)

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS - Claude Sonnet 4.5")
    print(f"{'='*70}")
    print(f"\nBehavioral Consistency (C):          {consistency:.3f}")
    print(f"Mean Performance (mu):                {mean_perf:.3f}")
    print(f"Std Performance (sigma):              {std_perf:.3f}")
    print(f"Context Propagation Effect (gamma):   {gamma:.3f}")

    print(f"\nMeasurement completed in {elapsed/60:.1f} minutes")

    # Generate code for registry
    print(f"\n{'='*70}")
    print("CODE FOR MODEL REGISTRY")
    print(f"{'='*70}")
    print(f'''
# Add to src/cert/models.py:

"claude-sonnet-4-20250514": ModelBaseline(
    model_id="claude-sonnet-4-20250514",
    provider="anthropic",
    model_family="Claude Sonnet 4.5",
    consistency={consistency:.3f},
    mean_performance={mean_perf:.3f},
    std_performance={std_perf:.3f},
    coordination_2agent={gamma:.3f},
    coordination_5agent=None,  # To be measured
    paper_section="Community Measurement",
    validation_date="2025-10",
),
''')

    # Comparison with GPT-5
    print(f"\n{'='*70}")
    print("COMPARISON WITH GPT-5")
    print(f"{'='*70}")

    gpt5_c = 0.702
    gpt5_mu = 0.543
    gpt5_sigma = 0.048
    gpt5_gamma = 1.911

    print(f"\n{'Metric':<30} {'Claude Sonnet 4.5':<20} {'GPT-5':<20} {'Difference':<15}")
    print("-"*85)
    print(f"{'Consistency (C)':<30} {consistency:<20.3f} {gpt5_c:<20.3f} {(consistency-gpt5_c)/gpt5_c*100:>+.1f}%")
    print(f"{'Mean Performance (mu)':<30} {mean_perf:<20.3f} {gpt5_mu:<20.3f} {(mean_perf-gpt5_mu)/gpt5_mu*100:>+.1f}%")
    print(f"{'Std Performance (sigma)':<30} {std_perf:<20.3f} {gpt5_sigma:<20.3f} {(std_perf-gpt5_sigma)/gpt5_sigma*100:>+.1f}%")
    print(f"{'Context Propagation (gamma)':<30} {gamma:<20.3f} {gpt5_gamma:<20.3f} {(gamma-gpt5_gamma)/gpt5_gamma*100:>+.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
