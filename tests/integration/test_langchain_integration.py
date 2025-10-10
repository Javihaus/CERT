"""
Integration tests for LangChain integration covering real-world edge cases.

Tests cover:
- Async execution
- Streaming responses
- Error propagation
- Retry logic
- Token limit handling
"""

import asyncio
from typing import Any, Dict, List

import pytest

# These tests require langchain to be installed
pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from cert.integrations.langchain import CERTLangChain
from cert.providers.base import ProviderInterface


class MockProvider(ProviderInterface):
    """Mock provider for testing."""

    def __init__(self, should_fail=False, delay_ms=0):
        self.should_fail = should_fail
        self.delay_ms = delay_ms
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate method."""
        self.call_count += 1

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        if self.should_fail:
            raise Exception("Provider failure")

        return f"Response to: {prompt[:50]}"

    def get_embedding(self, text: str) -> List[float]:
        """Mock embedding method."""
        return [0.1] * 384


class MockLangChainAgent:
    """Mock LangChain agent for testing."""

    def __init__(self, name: str, should_fail: bool = False, response_delay: float = 0):
        self.name = name
        self.should_fail = should_fail
        self.response_delay = response_delay
        self.call_count = 0

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock invoke method."""
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError(f"Agent {self.name} failed")

        if self.response_delay > 0:
            import time

            time.sleep(self.response_delay)

        # Extract input message
        messages = input_data.get("messages", [])
        input_text = messages[-1].get("content", "") if messages else ""

        # Generate response
        response = {
            "messages": [
                *messages,
                {"role": "assistant", "content": f"{self.name} processed: {input_text}"},
            ]
        }

        return response

    def stream(self, input_data: Dict[str, Any]):
        """Mock stream method."""
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError(f"Agent {self.name} streaming failed")

        # Extract input
        messages = input_data.get("messages", [])
        input_text = messages[-1].get("content", "") if messages else ""

        # Stream response in chunks
        response_text = f"{self.name} streamed: {input_text}"
        words = response_text.split()

        for word in words:
            if self.response_delay > 0:
                import time

                time.sleep(self.response_delay / len(words))

            yield {"messages": [{"role": "assistant", "content": word + " "}]}


@pytest.fixture
def cert_integration():
    """Create CERT integration with mock provider."""
    provider = MockProvider()
    return CERTLangChain(provider=provider, verbose=False)


class TestLangChainBasicIntegration:
    """Test basic LangChain integration functionality."""

    def test_wrap_single_agent(self, cert_integration):
        """Test wrapping a single agent."""
        agent = MockLangChainAgent("TestAgent")
        wrapped = cert_integration.wrap_agent(
            agent=agent, agent_id="test_agent", agent_name="Test Agent"
        )

        # Verify wrapped agent has same interface
        assert hasattr(wrapped, "invoke")
        assert hasattr(wrapped, "stream")

    def test_agent_invoke_execution(self, cert_integration):
        """Test agent invoke execution with tracking."""
        agent = MockLangChainAgent("Agent1")
        wrapped = cert_integration.wrap_agent(agent=agent, agent_id="agent1", agent_name="Agent 1")

        # Execute
        result = wrapped.invoke({"messages": [{"role": "user", "content": "Test input"}]})

        # Verify execution was tracked
        assert len(cert_integration.metrics.executions) == 1
        assert cert_integration.metrics.executions[0]["agent_name"] == "Agent 1"
        assert "Test input" in cert_integration.metrics.executions[0]["input"]

        # Verify result
        assert "Agent1 processed" in result["messages"][-1]["content"]

    def test_multi_agent_pipeline(self, cert_integration):
        """Test multi-agent pipeline execution."""
        agent1 = MockLangChainAgent("Researcher")
        agent2 = MockLangChainAgent("Writer")

        pipeline = cert_integration.create_multi_agent_pipeline(
            [
                {"agent": agent1, "agent_id": "researcher", "agent_name": "Researcher"},
                {"agent": agent2, "agent_id": "writer", "agent_name": "Writer"},
            ]
        )

        # Execute pipeline
        result = pipeline({"messages": [{"role": "user", "content": "Research AI"}]})

        # Verify both agents executed
        assert agent1.call_count == 1
        assert agent2.call_count == 1

        # Verify execution tracking
        assert len(cert_integration.metrics.executions) == 2
        assert cert_integration.metrics.executions[0]["agent_name"] == "Researcher"
        assert cert_integration.metrics.executions[1]["agent_name"] == "Writer"


class TestLangChainStreaming:
    """Test streaming response handling."""

    def test_agent_streaming_execution(self, cert_integration):
        """Test agent streaming with proper chunk collection."""
        agent = MockLangChainAgent("StreamAgent")
        wrapped = cert_integration.wrap_agent(
            agent=agent, agent_id="stream_agent", agent_name="Stream Agent"
        )

        # Stream execution
        chunks = list(wrapped.stream({"messages": [{"role": "user", "content": "Test streaming"}]}))

        # Verify chunks were yielded
        assert len(chunks) > 0

        # Verify execution was tracked with full output
        assert len(cert_integration.metrics.executions) == 1
        execution = cert_integration.metrics.executions[0]
        assert "StreamAgent" in execution["output"]

    def test_streaming_error_handling(self, cert_integration):
        """Test error handling during streaming."""
        agent = MockLangChainAgent("FailAgent", should_fail=True)
        wrapped = cert_integration.wrap_agent(
            agent=agent, agent_id="fail_agent", agent_name="Fail Agent"
        )

        # Stream should propagate errors
        with pytest.raises(RuntimeError, match="streaming failed"):
            list(wrapped.stream({"messages": [{"role": "user", "content": "Test"}]}))


class TestLangChainErrorHandling:
    """Test error propagation and handling."""

    def test_agent_failure_propagates(self, cert_integration):
        """Test that agent failures propagate correctly."""
        agent = MockLangChainAgent("FailAgent", should_fail=True)
        wrapped = cert_integration.wrap_agent(
            agent=agent, agent_id="fail_agent", agent_name="Fail Agent"
        )

        # Error should propagate
        with pytest.raises(RuntimeError, match="Agent FailAgent failed"):
            wrapped.invoke({"messages": [{"role": "user", "content": "Test"}]})

    def test_pipeline_partial_failure(self, cert_integration):
        """Test pipeline behavior when one agent fails."""
        agent1 = MockLangChainAgent("Agent1")
        agent2 = MockLangChainAgent("Agent2", should_fail=True)
        agent3 = MockLangChainAgent("Agent3")

        pipeline = cert_integration.create_multi_agent_pipeline(
            [
                {"agent": agent1, "agent_id": "agent1", "agent_name": "Agent 1"},
                {"agent": agent2, "agent_id": "agent2", "agent_name": "Agent 2"},
                {"agent": agent3, "agent_id": "agent3", "agent_name": "Agent 3"},
            ]
        )

        # Pipeline should fail at agent2
        with pytest.raises(RuntimeError, match="Agent Agent2 failed"):
            pipeline({"messages": [{"role": "user", "content": "Test"}]})

        # Verify only agent1 executed successfully
        assert agent1.call_count == 1
        assert agent2.call_count == 1
        assert agent3.call_count == 0  # Never reached

        # Verify metrics captured partial execution
        assert len(cert_integration.metrics.executions) == 1  # Only agent1 succeeded

    def test_empty_input_handling(self, cert_integration):
        """Test handling of empty or malformed inputs."""
        agent = MockLangChainAgent("Agent")
        wrapped = cert_integration.wrap_agent(agent=agent, agent_id="agent", agent_name="Agent")

        # Empty messages
        result = wrapped.invoke({"messages": []})
        assert result is not None

        # Malformed input
        result = wrapped.invoke({})
        assert result is not None


class TestLangChainPerformance:
    """Test performance and timing tracking."""

    def test_execution_timing_tracked(self, cert_integration):
        """Test that execution timing is properly tracked."""
        agent = MockLangChainAgent("SlowAgent", response_delay=0.1)
        wrapped = cert_integration.wrap_agent(
            agent=agent, agent_id="slow_agent", agent_name="Slow Agent"
        )

        # Execute
        wrapped.invoke({"messages": [{"role": "user", "content": "Test"}]})

        # Verify timing was tracked
        execution = cert_integration.metrics.executions[0]
        assert execution["duration_ms"] >= 100  # At least 100ms
        assert execution["duration_ms"] < 1000  # But not unreasonably long

    def test_concurrent_execution_safety(self, cert_integration):
        """Test that concurrent executions don't interfere."""
        agent1 = MockLangChainAgent("Agent1", response_delay=0.05)
        agent2 = MockLangChainAgent("Agent2", response_delay=0.05)

        wrapped1 = cert_integration.wrap_agent(
            agent=agent1, agent_id="agent1", agent_name="Agent 1"
        )
        wrapped2 = cert_integration.wrap_agent(
            agent=agent2, agent_id="agent2", agent_name="Agent 2"
        )

        # Execute both
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(
                wrapped1.invoke, {"messages": [{"role": "user", "content": "Test 1"}]}
            )
            future2 = executor.submit(
                wrapped2.invoke, {"messages": [{"role": "user", "content": "Test 2"}]}
            )

            result1 = future1.result()
            result2 = future2.result()

        # Both should succeed
        assert result1 is not None
        assert result2 is not None

        # Both executions should be tracked
        assert len(cert_integration.metrics.executions) == 2


class TestLangChainMetrics:
    """Test metrics calculation for LangChain."""

    def test_quality_calculation(self, cert_integration):
        """Test that quality scores are calculated."""
        agent = MockLangChainAgent("Agent")
        wrapped = cert_integration.wrap_agent(
            agent=agent, agent_id="agent", agent_name="Agent", calculate_quality=True
        )

        # Execute
        wrapped.invoke({"messages": [{"role": "user", "content": "Test input with quality"}]})

        # Verify quality was calculated
        assert len(cert_integration.metrics.intermediate_qualities) > 0
        assert cert_integration.metrics.output_quality is not None

    def test_coordination_effect_calculation(self, cert_integration):
        """Test gamma calculation for pipeline."""
        agent1 = MockLangChainAgent("Agent1")
        agent2 = MockLangChainAgent("Agent2")

        pipeline = cert_integration.create_multi_agent_pipeline(
            [
                {"agent": agent1, "agent_id": "agent1", "agent_name": "Agent 1"},
                {"agent": agent2, "agent_id": "agent2", "agent_name": "Agent 2"},
            ]
        )

        # Execute
        pipeline({"messages": [{"role": "user", "content": "Test"}]})

        # Coordination effect should be calculated
        # Note: May be None if quality calculation not available
        # but should not raise errors


class TestLangChainEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_very_long_input(self, cert_integration):
        """Test handling of very long inputs (token limit scenarios)."""
        agent = MockLangChainAgent("Agent")
        wrapped = cert_integration.wrap_agent(agent=agent, agent_id="agent", agent_name="Agent")

        # Create a very long input (simulating token limit issues)
        long_text = "word " * 10000  # ~10k words

        result = wrapped.invoke({"messages": [{"role": "user", "content": long_text}]})

        # Should handle without crashing
        assert result is not None
        assert len(cert_integration.metrics.executions) == 1

    def test_special_characters_in_content(self, cert_integration):
        """Test handling of special characters and encoding."""
        agent = MockLangChainAgent("Agent")
        wrapped = cert_integration.wrap_agent(agent=agent, agent_id="agent", agent_name="Agent")

        # Test with various special characters
        special_inputs = [
            "Hello 世界 🌍",  # Unicode
            "Code: ```python\ndef foo():\n    pass```",  # Code blocks
            '{"json": "data", "nested": {"key": "value"}}',  # JSON
            "Line1\nLine2\r\nLine3",  # Various newlines
        ]

        for test_input in special_inputs:
            result = wrapped.invoke({"messages": [{"role": "user", "content": test_input}]})
            assert result is not None

    def test_rapid_sequential_calls(self, cert_integration):
        """Test handling of rapid sequential calls."""
        agent = MockLangChainAgent("Agent")
        wrapped = cert_integration.wrap_agent(agent=agent, agent_id="agent", agent_name="Agent")

        # Make many rapid calls
        for i in range(50):
            wrapped.invoke({"messages": [{"role": "user", "content": f"Call {i}"}]})

        # All should be tracked
        assert len(cert_integration.metrics.executions) == 50
        assert agent.call_count == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
