"""Unit tests for node functions."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.unit
@pytest.mark.asyncio
class TestAnalystNodes:
    """Tests for analyst generation nodes."""

    async def test_create_analysts(self, mock_llm, mock_mcp_client, monkeypatch, sample_analysts):
        """Test create_analysts node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from module4.model import Perspectives

        # Mock MCP client
        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Mock structured LLM output
        perspectives = Perspectives(analysts=sample_analysts)
        mock_llm.with_structured_output().ainvoke = AsyncMock(return_value=perspectives)

        state = {
            "topic": "AI",
            "max_analysts": 3,
            "human_analyst_feedback": ""
        }

        result = await research_assistant.create_analysts(state, mock_llm)

        assert "analysts" in result
        assert len(result["analysts"]) == 3
        assert result["analysts"][0].name == "Dr. Jane Smith"

    def test_human_feedback(self, sample_generate_analysts_state):
        """Test human_feedback node (no-op)."""
        from module4 import research_assistant

        # Should not raise any errors
        result = research_assistant.human_feedback(sample_generate_analysts_state)
        assert result is None

    def test_should_continue_with_feedback(self):
        """Test should_continue returns create_analysts when feedback exists."""
        from module4 import research_assistant
        from langgraph.constants import END

        state = {
            "topic": "AI",
            "max_analysts": 3,
            "human_analyst_feedback": "Focus more on ethics",
            "analysts": []
        }

        result = research_assistant.should_continue(state)
        assert result == "create_analysts"

    def test_should_continue_without_feedback(self):
        """Test should_continue returns END when no feedback."""
        from module4 import research_assistant
        from langgraph.constants import END

        state = {
            "topic": "AI",
            "max_analysts": 3,
            "human_analyst_feedback": None,
            "analysts": []
        }

        result = research_assistant.should_continue(state)
        assert result == END


@pytest.mark.unit
@pytest.mark.asyncio
class TestInterviewNodes:
    """Tests for interview nodes."""

    async def test_generate_question(self, mock_llm, mock_mcp_client, monkeypatch, sample_interview_state):
        """Test generate_question node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="What is AI?"))

        result = await research_assistant.generate_question(sample_interview_state, mock_llm)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "What is AI?"

    async def test_search_web(self, mock_llm, mock_tavily_search, mock_mcp_client, monkeypatch, sample_interview_state):
        """Test search_web node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from module4.model import SearchQuery

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Mock structured output
        query = SearchQuery(search_query="artificial intelligence")
        mock_llm.with_structured_output().ainvoke = AsyncMock(return_value=query)

        result = await research_assistant.search_web(sample_interview_state, mock_llm, mock_tavily_search)

        assert "context" in result
        assert len(result["context"]) == 1
        assert "example.com" in result["context"][0]

    async def test_generate_answer(self, mock_llm, mock_mcp_client, monkeypatch, sample_interview_state):
        """Test generate_answer node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        sample_interview_state["context"] = ["Test context"]
        mock_answer = AIMessage(content="AI is artificial intelligence")
        mock_llm.ainvoke = AsyncMock(return_value=mock_answer)

        result = await research_assistant.generate_answer(sample_interview_state, mock_llm)

        assert "messages" in result
        assert result["messages"][0].name == "expert"
        assert "artificial intelligence" in result["messages"][0].content

    def test_save_interview(self, sample_interview_state):
        """Test save_interview node."""
        from module4 import research_assistant

        sample_interview_state["messages"] = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1")
        ]

        result = research_assistant.save_interview(sample_interview_state)

        assert "interview" in result
        assert "Question 1" in result["interview"]
        assert "Answer 1" in result["interview"]

    def test_route_messages_continue(self, sample_interview_state):
        """Test route_messages returns ask_question."""
        from module4 import research_assistant

        sample_interview_state["messages"] = [
            HumanMessage(content="Question"),
            AIMessage(content="Answer", name="expert")
        ]
        sample_interview_state["max_num_turns"] = 5

        result = research_assistant.route_messages(sample_interview_state)
        assert result == "ask_question"

    def test_route_messages_end_max_turns(self, sample_interview_state):
        """Test route_messages returns save_interview when max turns reached."""
        from module4 import research_assistant

        sample_interview_state["messages"] = [
            AIMessage(content="Answer 1", name="expert"),
            AIMessage(content="Answer 2", name="expert")
        ]
        sample_interview_state["max_num_turns"] = 2

        result = research_assistant.route_messages(sample_interview_state)
        assert result == "save_interview"

    def test_route_messages_end_thank_you(self, sample_interview_state):
        """Test route_messages detects 'Thank you' message."""
        from module4 import research_assistant

        sample_interview_state["messages"] = [
            HumanMessage(content="Thank you so much for your help!"),
            AIMessage(content="You're welcome")
        ]

        result = research_assistant.route_messages(sample_interview_state)
        assert result == "save_interview"

    async def test_write_section(self, mock_llm, mock_mcp_client, monkeypatch, sample_interview_state):
        """Test write_section node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        sample_interview_state["interview"] = "Interview transcript"
        sample_interview_state["context"] = ["Context 1"]

        mock_section = AIMessage(content="## Section Title\n\nSection content")
        mock_llm.ainvoke = AsyncMock(return_value=mock_section)

        result = await research_assistant.write_section(sample_interview_state, mock_llm)

        assert "sections" in result
        assert len(result["sections"]) == 1
        assert "Section Title" in result["sections"][0]


@pytest.mark.unit
@pytest.mark.asyncio
class TestReportNodes:
    """Tests for report generation nodes."""

    def test_initiate_all_interviews(self, sample_research_graph_state):
        """Test initiate_all_interviews returns Send objects."""
        from module4 import research_assistant
        from langgraph.types import Send

        result = research_assistant.initiate_all_interviews(sample_research_graph_state)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, Send) for item in result)

    def test_initiate_all_interviews_with_feedback(self):
        """Test initiate_all_interviews with feedback returns create_analysts."""
        from module4 import research_assistant

        state = {
            "topic": "AI",
            "max_analysts": 3,
            "human_analyst_feedback": "More technical",
            "analysts": []
        }

        result = research_assistant.initiate_all_interviews(state)
        assert result == "create_analysts"

    async def test_write_report(self, mock_llm, mock_mcp_client, monkeypatch, sample_research_graph_state):
        """Test write_report node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        sample_research_graph_state["sections"] = ["Section 1", "Section 2"]

        mock_report = AIMessage(content="## Insights\n\nReport content")
        mock_llm.ainvoke = AsyncMock(return_value=mock_report)

        result = await research_assistant.write_report(sample_research_graph_state, mock_llm)

        assert "content" in result
        assert "Insights" in result["content"]

    async def test_write_introduction(self, mock_llm, mock_mcp_client, monkeypatch, sample_research_graph_state):
        """Test write_introduction node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        sample_research_graph_state["sections"] = ["Section 1"]

        mock_intro = AIMessage(content="# Report Title\n\n## Introduction\n\nIntro content")
        mock_llm.ainvoke = AsyncMock(return_value=mock_intro)

        result = await research_assistant.write_introduction(sample_research_graph_state, mock_llm)

        assert "introduction" in result
        assert "Introduction" in result["introduction"]

    async def test_write_conclusion(self, mock_llm, mock_mcp_client, monkeypatch, sample_research_graph_state):
        """Test write_conclusion node."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        sample_research_graph_state["sections"] = ["Section 1"]

        mock_conclusion = AIMessage(content="## Conclusion\n\nConclusion content")
        mock_llm.ainvoke = AsyncMock(return_value=mock_conclusion)

        result = await research_assistant.write_conclusion(sample_research_graph_state, mock_llm)

        assert "conclusion" in result
        assert "Conclusion" in result["conclusion"]

    def test_finalize_report(self):
        """Test finalize_report combines all sections."""
        from module4 import research_assistant

        state = {
            "introduction": "# Title\n\n## Introduction\n\nIntro",
            "content": "## Insights\n\nContent\n\n## Sources\n[1] Source 1",
            "conclusion": "## Conclusion\n\nConclusion",
            "sections": []
        }

        result = research_assistant.finalize_report(state)

        assert "final_report" in result
        assert "Introduction" in result["final_report"]
        assert "Content" in result["final_report"]
        assert "Conclusion" in result["final_report"]
        assert "Sources" in result["final_report"]
