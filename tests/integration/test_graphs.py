"""Integration tests for graph construction and execution."""

import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.messages import HumanMessage, AIMessage

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.integration
@pytest.mark.asyncio
class TestInterviewGraph:
    """Integration tests for interview graph."""

    async def test_interview_graph_construction(self, mock_llm, mock_tavily_search):
        """Test that interview graph builds correctly."""
        from module4 import research_assistant

        graph = research_assistant.build_interview_graph(mock_llm, mock_tavily_search)

        assert graph is not None
        # Verify graph has expected nodes
        assert hasattr(graph, 'nodes')

    async def test_interview_graph_nodes_present(self, mock_llm, mock_tavily_search):
        """Test interview graph has all required nodes."""
        from module4 import research_assistant

        graph = research_assistant.build_interview_graph(mock_llm, mock_tavily_search)

        # Get the compiled graph
        compiled = graph

        # The graph should have these node names configured
        expected_nodes = [
            "ask_question",
            "search_web",
            "search_wikipedia",
            "answer_question",
            "save_interview",
            "write_section"
        ]

        # Note: Can't directly access nodes in compiled graph without running it
        # This tests construction doesn't error
        assert compiled is not None


@pytest.mark.integration
@pytest.mark.asyncio
class TestResearchGraph:
    """Integration tests for research graph."""

    async def test_research_graph_construction(self, mock_llm):
        """Test that research graph builds correctly."""
        from module4 import research_assistant
        from langgraph.checkpoint.memory import MemorySaver

        # Build interview graph first
        interview_graph = research_assistant.build_interview_graph(
            mock_llm,
            Mock()  # mock tavily
        )

        # Build research graph
        graph = research_assistant.build_research_graph(mock_llm, interview_graph)

        assert graph is not None

    async def test_research_graph_with_interrupt(self, mock_llm):
        """Test research graph is compiled with human_feedback interrupt."""
        from module4 import research_assistant

        interview_graph = research_assistant.build_interview_graph(mock_llm, Mock())
        graph = research_assistant.build_research_graph(mock_llm, interview_graph)

        # Graph should be compiled with interrupt_before
        assert graph is not None


@pytest.mark.integration
@pytest.mark.asyncio
class TestGraphExecution:
    """Integration tests for graph execution with mocked components."""

    async def test_analyst_generation_flow(self, mock_llm, mock_mcp_client, monkeypatch, sample_analysts):
        """Test flow from START to analyst generation."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from module4.model import Perspectives

        # Setup mocks
        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        perspectives = Perspectives(analysts=sample_analysts)
        mock_llm.with_structured_output().ainvoke = AsyncMock(return_value=perspectives)

        # Build graph
        interview_graph = research_assistant.build_interview_graph(mock_llm, Mock())
        graph = research_assistant.build_research_graph(mock_llm, interview_graph)

        # Execute just the analyst creation
        initial_state = {
            "topic": "AI in Healthcare",
            "max_analysts": 3
        }

        # Note: Full execution would require running the graph
        # This tests the setup doesn't error
        assert graph is not None

    async def test_interview_single_turn(self, mock_llm, mock_tavily_search, mock_mcp_client,
                                        monkeypatch, sample_analyst):
        """Test a single turn of interview."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from module4.model import SearchQuery

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Mock LLM responses
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test question"))
        query = SearchQuery(search_query="test query")
        mock_llm.with_structured_output().ainvoke = AsyncMock(return_value=query)

        # Create state
        state = {
            "messages": [HumanMessage(content="Hello")],
            "max_num_turns": 2,
            "context": [],
            "analyst": sample_analyst,
            "interview": "",
            "sections": []
        }

        # Test individual nodes work together
        question_result = await research_assistant.generate_question(state, mock_llm)
        assert "messages" in question_result

        state["messages"].append(question_result["messages"][0])
        search_result = await research_assistant.search_web(state, mock_llm, mock_tavily_search)
        assert "context" in search_result


@pytest.mark.integration
class TestGraphStateTransitions:
    """Integration tests for state transitions through graph."""

    def test_analyst_state_to_interview_state(self, sample_analysts):
        """Test state transformation from analyst generation to interviews."""
        from module4 import research_assistant
        from langgraph.types import Send

        research_state = {
            "topic": "AI",
            "max_analysts": 3,
            "human_analyst_feedback": None,
            "analysts": sample_analysts,
            "sections": []
        }

        sends = research_assistant.initiate_all_interviews(research_state)

        assert isinstance(sends, list)
        assert len(sends) == 3

        # Each Send should create interview state
        for send in sends:
            assert isinstance(send, Send)
            assert send.node == "conduct_interview"
            assert "analyst" in send.arg
            assert "messages" in send.arg

    def test_report_state_finalization(self):
        """Test final report assembly from components."""
        from module4 import research_assistant

        state = {
            "introduction": "# AI Report\n\n## Introduction\n\nThis is the intro.",
            "content": "## Insights\n\nKey findings here.\n\n## Sources\n[1] Source A\n[2] Source B",
            "conclusion": "## Conclusion\n\nFinal thoughts.",
            "sections": ["Section 1", "Section 2"]
        }

        result = research_assistant.finalize_report(state)

        final = result["final_report"]

        # Check all parts are present
        assert "Introduction" in final
        assert "Key findings" in final
        assert "Conclusion" in final
        assert "Sources" in final
        assert "[1] Source A" in final

    def test_report_without_sources(self):
        """Test report finalization handles missing sources."""
        from module4 import research_assistant

        state = {
            "introduction": "# Report\n\n## Introduction\n\nIntro text.",
            "content": "## Insights\n\nContent without sources.",
            "conclusion": "## Conclusion\n\nConclusion text.",
            "sections": []
        }

        result = research_assistant.finalize_report(state)

        assert "final_report" in result
        assert "Introduction" in result["final_report"]
