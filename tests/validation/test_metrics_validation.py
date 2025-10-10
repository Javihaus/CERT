"""
Validation tests proving CERT metric implementations match paper formulas.

These tests verify that the implementation exactly matches the mathematical
definitions from the paper using known test cases.
"""

import pytest
import numpy as np
from cert.core.metrics import (
    behavioral_consistency,
    empirical_performance_distribution,
    coordination_effect,
    performance_baseline,
    prediction_error,
    pipeline_health_score,
    performance_variability,
)


class TestBehavioralConsistencyValidation:
    """Validate behavioral consistency C = 1 - (σ(d) / μ(d))."""

    def test_matches_paper_formula(self):
        """Verify C calculation matches paper Equation 1."""
        # Example from paper validation
        distances = np.array([0.12, 0.15, 0.13, 0.14, 0.16, 0.12, 0.15])

        # Manual calculation per paper formula
        std = np.std(distances, ddof=1)
        mean = np.mean(distances)
        expected = 1.0 - (std / mean)

        # SDK calculation
        actual = behavioral_consistency(distances)

        # Should match within floating point precision
        assert abs(actual - expected) < 1e-10

    def test_high_consistency_scenario(self):
        """Test high consistency (low variance) scenario."""
        # Very similar distances → high consistency
        distances = np.array([0.80, 0.82, 0.81, 0.83, 0.80])

        c = behavioral_consistency(distances)

        # Should be high (approaching 1.0)
        assert c > 0.95
        assert c <= 1.0

    def test_low_consistency_scenario(self):
        """Test low consistency (high variance) scenario."""
        # Widely varying distances → low consistency
        distances = np.array([0.1, 0.5, 0.9, 0.2, 0.8])

        c = behavioral_consistency(distances)

        # Should be low
        assert c < 0.7
        assert c >= 0.0

    def test_perfect_consistency(self):
        """Test perfect consistency (all identical)."""
        distances = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        c = behavioral_consistency(distances)

        # Perfect consistency → C = 1.0
        assert abs(c - 1.0) < 0.001

    def test_paper_example_values(self):
        """Test with values from paper Table 2."""
        # GPT-4o example: C = 0.831
        # Simulated distances that would produce this C
        mean_dist = 0.15
        target_c = 0.831
        # From C = 1 - (σ/μ), we get σ = μ(1-C)
        std_dist = mean_dist * (1 - target_c)

        # Generate distances with this mean and std
        np.random.seed(42)
        distances = np.random.normal(mean_dist, std_dist, 20)
        distances = np.abs(distances)  # Ensure positive

        c = behavioral_consistency(distances)

        # Should be close to target
        assert abs(c - target_c) < 0.1  # Within 10%


class TestEmpiricalPerformanceDistribution:
    """Validate empirical performance distribution (μ, σ)."""

    def test_matches_paper_formula(self):
        """Verify (μ, σ) calculation matches paper Equation 2."""
        scores = np.array([0.85, 0.82, 0.88, 0.79, 0.86, 0.83, 0.87])

        # Manual calculation
        expected_mu = np.mean(scores)
        expected_sigma = np.std(scores, ddof=1)  # Sample std

        # SDK calculation
        actual_mu, actual_sigma = empirical_performance_distribution(scores)

        # Should match exactly
        assert abs(actual_mu - expected_mu) < 1e-10
        assert abs(actual_sigma - expected_sigma) < 1e-10

    def test_paper_baseline_values(self):
        """Test with baseline values from paper."""
        # Claude Sonnet 4.5: μ=0.745, σ=0.058
        # Generate scores matching this distribution
        np.random.seed(42)
        scores = np.random.normal(0.745, 0.058, 100)
        scores = np.clip(scores, 0, 1)  # Ensure valid range

        mu, sigma = empirical_performance_distribution(scores)

        # Should be close to target
        assert abs(mu - 0.745) < 0.05
        assert abs(sigma - 0.058) < 0.02


class TestCoordinationEffectValidation:
    """Validate coordination effect γ = P_obs / baseline."""

    def test_corrected_formula(self):
        """Verify corrected γ calculation uses baseline, not product."""
        # Test case from the bug fix
        coordinated = 0.460
        independent = [0.489, 0.397, 0.460]

        # Correct calculation: divide by mean baseline
        expected_baseline = np.mean(independent)
        expected_gamma = coordinated / expected_baseline

        # SDK calculation
        actual_gamma = coordination_effect(coordinated, independent)

        # Should match
        assert abs(actual_gamma - expected_gamma) < 1e-10

        # Verify this gives the correct value (0.847)
        assert abs(actual_gamma - 0.847) < 0.001

    def test_no_context_benefit(self):
        """Test γ = 1.0 when sequential equals baseline."""
        # No benefit from sequential processing
        coordinated = 0.75
        independent = [0.75, 0.75]

        gamma = coordination_effect(coordinated, independent)

        # Should be exactly 1.0
        assert abs(gamma - 1.0) < 1e-10

    def test_positive_context_effect(self):
        """Test γ > 1.0 when sequential improves performance."""
        # Sequential processing helps
        coordinated = 0.80
        independent = [0.65, 0.70]  # Mean = 0.675

        gamma = coordination_effect(coordinated, independent)

        # Should be > 1.0
        assert gamma > 1.0
        assert abs(gamma - (0.80 / 0.675)) < 1e-10

    def test_negative_context_effect(self):
        """Test γ < 1.0 when sequential degrades performance."""
        # Sequential processing hurts (context dilution)
        coordinated = 0.60
        independent = [0.75, 0.80]  # Mean = 0.775

        gamma = coordination_effect(coordinated, independent)

        # Should be < 1.0
        assert gamma < 1.0
        assert abs(gamma - (0.60 / 0.775)) < 1e-10

    def test_paper_gamma_values(self):
        """Test with γ values from paper Table 2."""
        # GPT-5: γ = 1.911 (strongest context effect)
        # If baseline is 0.543 and γ = 1.911:
        # coordinated = γ × baseline
        baseline_perf = 0.543
        target_gamma = 1.911
        coordinated = target_gamma * baseline_perf

        gamma = coordination_effect(coordinated, [baseline_perf, baseline_perf])

        # Should match target
        assert abs(gamma - target_gamma) < 0.001


class TestPerformanceBaselineValidation:
    """Validate performance baseline P_baseline from Equation 5."""

    def test_matches_paper_formula(self):
        """Verify baseline calculation matches paper Equation 5."""
        # Example: 3-agent pipeline
        agent_means = np.array([0.65, 0.70, 0.68])
        gamma_mean = 1.35
        alpha = 0.93
        n = len(agent_means)

        # Manual calculation per paper
        mu_bar = np.mean(agent_means)
        coord_adj = 1.0 + (gamma_mean - 1.0) / np.sqrt(n - 1)
        phi_n = alpha ** (n - 1)
        expected = mu_bar * coord_adj * phi_n

        # SDK calculation
        actual = performance_baseline(agent_means, gamma_mean, alpha)

        # Should match exactly
        assert abs(actual - expected) < 1e-10

    def test_two_agent_pipeline(self):
        """Test baseline for two-agent pipeline (special case)."""
        agent_means = np.array([0.65, 0.70])
        gamma_mean = 1.40
        alpha = 0.93

        baseline = performance_baseline(agent_means, gamma_mean, alpha)

        # For n=2, sqrt(n-1) = 1.0, alpha^1 = 0.93
        mu_bar = 0.675
        expected = mu_bar * gamma_mean * alpha

        assert abs(baseline - expected) < 1e-10


class TestPredictionErrorValidation:
    """Validate prediction error ε from Equation 6."""

    def test_matches_paper_formula(self):
        """Verify ε calculation matches paper Equation 6."""
        observed = 0.72
        baseline = 0.65

        # Manual calculation
        expected = abs(observed - baseline) / baseline

        # SDK calculation
        actual = prediction_error(observed, baseline)

        # Should match exactly
        assert abs(actual - expected) < 1e-10

    def test_perfect_prediction(self):
        """Test ε = 0 for perfect prediction."""
        observed = 0.75
        baseline = 0.75

        epsilon = prediction_error(observed, baseline)

        assert epsilon == 0.0

    def test_gpt4_low_error(self):
        """Test with GPT-4's near-zero error from paper."""
        # GPT-4: ε = 0.003 (best prediction accuracy)
        baseline = 0.638
        # With ε = 0.003: observed = baseline × (1 + ε)
        observed = baseline * 1.003

        epsilon = prediction_error(observed, baseline)

        assert abs(epsilon - 0.003) < 0.0001


class TestPipelineHealthScoreValidation:
    """Validate pipeline health score H from Equation 7."""

    def test_matches_paper_formula(self):
        """Verify H calculation matches paper Equation 7."""
        epsilon = 0.05
        gamma_mean = 1.35
        cov = 0.95

        # Manual calculation per paper
        accuracy_factor = 1.0 / (1.0 + epsilon)
        context_factor = min(1.0, gamma_mean)
        expected = accuracy_factor * context_factor * cov

        # SDK calculation
        actual = pipeline_health_score(epsilon, gamma_mean, cov)

        # Should match exactly
        assert abs(actual - expected) < 1e-10

    def test_healthy_pipeline_scenario(self):
        """Test healthy pipeline (H > 0.80)."""
        # Low error, good context effect, high coverage
        epsilon = 0.05
        gamma_mean = 1.35
        cov = 0.95

        health = pipeline_health_score(epsilon, gamma_mean, cov)

        # Should indicate healthy pipeline
        assert health > 0.80

    def test_degraded_pipeline_scenario(self):
        """Test degraded pipeline (H < 0.60)."""
        # High error, weak context effect, low coverage
        epsilon = 0.50
        gamma_mean = 0.85
        cov = 0.75

        health = pipeline_health_score(epsilon, gamma_mean, cov)

        # Should indicate degraded pipeline
        assert health < 0.60

    def test_gamma_capping(self):
        """Test that γ > 1 is capped at 1.0 for stability."""
        epsilon = 0.10
        gamma_mean = 1.80  # High context effect
        cov = 1.0

        health = pipeline_health_score(epsilon, gamma_mean, cov)

        # γ should be capped at 1.0
        # H = (1/1.1) × 1.0 × 1.0 = 0.909
        expected = (1.0 / 1.1) * 1.0 * 1.0
        assert abs(health - expected) < 1e-10


class TestPerformanceVariabilityValidation:
    """Validate performance variability Ω."""

    def test_matches_formula(self):
        """Verify Ω = σ_obs / σ_baseline."""
        observed_std = 0.12
        baseline_std = 0.05

        expected = observed_std / baseline_std

        actual = performance_variability(observed_std, baseline_std)

        assert abs(actual - expected) < 1e-10

    def test_paper_variability_values(self):
        """Test with Ω values from paper Table 5."""
        # GPT-4: Ω = 2.320 (moderate variability)
        # If baseline σ = 0.069, then observed σ = 0.069 × 2.320
        baseline_std = 0.069
        target_omega = 2.320
        observed_std = baseline_std * target_omega

        omega = performance_variability(observed_std, baseline_std)

        assert abs(omega - target_omega) < 0.001

    def test_low_variability(self):
        """Test low variability scenario (Ω < 2.0)."""
        # Gemini: Ω = 1.729 (stable)
        observed_std = 0.090
        baseline_std = 0.052  # 0.090 / 1.729

        omega = performance_variability(observed_std, baseline_std)

        assert omega < 2.0
        assert abs(omega - 1.729) < 0.01

    def test_high_variability(self):
        """Test high variability scenario (Ω >= 2.5)."""
        # Claude: Ω = 2.852 (requires intensive monitoring)
        observed_std = 0.075
        baseline_std = 0.026  # 0.075 / 2.852

        omega = performance_variability(observed_std, baseline_std)

        assert omega >= 2.5
        assert abs(omega - 2.852) < 0.01


class TestFormulaCross Validation:
    """Cross-validate formulas against each other."""

    def test_baseline_and_error_consistency(self):
        """Test that baseline and error calculations are consistent."""
        # If we predict a baseline and measure actual performance,
        # the error should be calculable both ways
        agent_means = np.array([0.65, 0.70, 0.68])
        gamma_mean = 1.35
        alpha = 0.93

        predicted_baseline = performance_baseline(agent_means, gamma_mean, alpha)

        # Simulate actual observed performance
        actual_observed = predicted_baseline * 1.10  # 10% better than predicted

        # Calculate error
        epsilon = prediction_error(actual_observed, predicted_baseline)

        # Error should be 10%
        assert abs(epsilon - 0.10) < 0.001

    def test_health_score_components(self):
        """Test that health score properly integrates all components."""
        # Create a scenario with known values
        epsilon = 0.15
        gamma_mean = 1.25
        cov = 0.90

        health = pipeline_health_score(epsilon, gamma_mean, cov)

        # Verify each component contribution
        accuracy_contribution = 1.0 / (1.0 + epsilon)  # ~0.870
        context_contribution = min(1.0, gamma_mean)  # 1.0 (capped)
        coverage_contribution = cov  # 0.90

        expected = accuracy_contribution * context_contribution * coverage_contribution

        assert abs(health - expected) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
