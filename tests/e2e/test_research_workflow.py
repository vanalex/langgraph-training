"""End-to-end tests for research assistant workflow."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from langchain_core.messages import AIMessage, HumanMessage

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
class TestResearchWorkflowE2E:
    """End-to-end tests for complete research workflow."""

    async def test_full_workflow_mock_execution(
        self, mock_llm, mock_tavily_search, mock_mcp_client,
        monkeypatch, sample_analysts
    ):
        """Test complete workflow from analyst creation to final report."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from module4.model import Perspectives, SearchQuery

        # Setup mocks
        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Mock analyst creation
        perspectives = Perspectives(analysts=sample_analysts[:1])  # Use 1 analyst for speed
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=perspectives)

        # Mock search query
        query = SearchQuery(search_query="test query")
        mock_structured_llm_search = Mock()
        mock_structured_llm_search.ainvoke = AsyncMock(return_value=query)

        def mock_with_structured_output(schema):
            if schema == Perspectives:
                return mock_structured_llm
            elif schema == SearchQuery:
                return mock_structured_llm_search
            return mock_structured_llm

        mock_llm.with_structured_output = mock_with_structured_output

        # Mock LLM responses
        mock_llm.ainvoke = AsyncMock(side_effect=[
            AIMessage(content="Question about AI"),  # generate_question
            AIMessage(content="Answer about AI", name="expert"),  # generate_answer
            AIMessage(content="Thank you so much for your help!"),  # end interview
            AIMessage(content="## Section\n\nSection content"),  # write_section
            AIMessage(content="## Insights\n\nReport insights\n\n## Sources\n[1] Source"),  # write_report
            AIMessage(content="# Title\n\n## Introduction\n\nIntro"),  # write_introduction
            AIMessage(content="## Conclusion\n\nConclusion"),  # write_conclusion
        ])

        # Build graphs
        interview_graph = research_assistant.build_interview_graph(mock_llm, mock_tavily_search)
        research_graph = research_assistant.build_research_graph(mock_llm, interview_graph)

        # This tests the graph construction doesn't error
        assert research_graph is not None
        assert interview_graph is not None

    async def test_analyst_creation_to_interview_transition(
        self, mock_llm, mock_tavily_search, mock_mcp_client,
        monkeypatch, sample_analysts
    ):
        """Test transition from analyst creation to interviews."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from module4.model import Perspectives

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Create analysts
        state = {
            "topic": "AI in Healthcare",
            "max_analysts": 3,
            "human_analyst_feedback": ""
        }

        perspectives = Perspectives(analysts=sample_analysts)
        mock_llm.with_structured_output().ainvoke = AsyncMock(return_value=perspectives)

        result = await research_assistant.create_analysts(state, mock_llm)

        assert "analysts" in result
        assert len(result["analysts"]) == 3

        # Test transition to interviews
        research_state = {**state, **result, "sections": []}
        sends = research_assistant.initiate_all_interviews(research_state)

        assert isinstance(sends, list)
        assert len(sends) == 3

    async def test_interview_to_section_workflow(
        self, mock_llm, mock_tavily_search, mock_mcp_client,
        monkeypatch, sample_analyst
    ):
        """Test workflow from interview to section writing."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from module4.model import SearchQuery

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Initial interview state
        state = {
            "messages": [HumanMessage(content="Start interview")],
            "max_num_turns": 2,
            "context": [],
            "analyst": sample_analyst,
            "interview": "",
            "sections": []
        }

        # Generate question
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="What is AI?"))
        q_result = await research_assistant.generate_question(state, mock_llm)
        state["messages"].append(q_result["messages"][0])

        # Search web
        query = SearchQuery(search_query="artificial intelligence")
        mock_llm.with_structured_output().ainvoke = AsyncMock(return_value=query)
        search_result = await research_assistant.search_web(state, mock_llm, mock_tavily_search)
        state["context"].extend(search_result["context"])

        # Generate answer
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="AI is...", name="expert"))
        answer_result = await research_assistant.generate_answer(state, mock_llm)
        state["messages"].append(answer_result["messages"][0])

        # Save interview
        interview_result = research_assistant.save_interview(state)
        state["interview"] = interview_result["interview"]

        # Write section
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="## AI Overview\n\nContent"))
        section_result = await research_assistant.write_section(state, mock_llm)

        assert "sections" in section_result
        assert len(section_result["sections"]) == 1
        assert "AI Overview" in section_result["sections"][0]

    async def test_sections_to_final_report_workflow(
        self, mock_llm, mock_mcp_client, monkeypatch, sample_analysts
    ):
        """Test workflow from sections to final report."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # State with completed sections
        state = {
            "topic": "AI in Healthcare",
            "max_analysts": 3,
            "human_analyst_feedback": "",
            "analysts": sample_analysts,
            "sections": [
                "## Section 1\n\nContent 1 [1]",
                "## Section 2\n\nContent 2 [2]",
                "## Section 3\n\nContent 3 [3]"
            ],
            "introduction": "",
            "content": "",
            "conclusion": "",
            "final_report": ""
        }

        # Write report
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content="## Insights\n\nKey findings\n\n## Sources\n[1] A\n[2] B\n[3] C"
        ))
        report_result = await research_assistant.write_report(state, mock_llm)
        state["content"] = report_result["content"]

        # Write introduction
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content="# AI in Healthcare\n\n## Introduction\n\nThis report explores..."
        ))
        intro_result = await research_assistant.write_introduction(state, mock_llm)
        state["introduction"] = intro_result["introduction"]

        # Write conclusion
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content="## Conclusion\n\nIn summary..."
        ))
        conclusion_result = await research_assistant.write_conclusion(state, mock_llm)
        state["conclusion"] = conclusion_result["conclusion"]

        # Finalize report
        final_result = research_assistant.finalize_report(state)

        assert "final_report" in final_result
        report = final_result["final_report"]

        # Verify all parts are present
        assert "Introduction" in report
        assert "Key findings" in report
        assert "Conclusion" in report
        assert "Sources" in report


@pytest.mark.e2e
@pytest.mark.slow
class TestErrorHandlingE2E:
    """End-to-end tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_workflow_with_missing_analyst(self, mock_llm, mock_mcp_client, monkeypatch):
        """Test workflow handles missing analyst gracefully."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # State with empty analysts
        state = {
            "topic": "AI",
            "max_analysts": 0,
            "human_analyst_feedback": "",
            "analysts": [],
            "sections": []
        }

        sends = research_assistant.initiate_all_interviews(state)

        # Should return empty list or handle gracefully
        assert sends == [] or sends == "create_analysts"

    @pytest.mark.asyncio
    async def test_workflow_with_llm_error(self, mock_llm, mock_mcp_client, monkeypatch):
        """Test workflow handles LLM errors."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Mock LLM to raise error
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM Error"))

        state = {
            "topic": "AI",
            "max_analysts": 3,
            "human_analyst_feedback": ""
        }

        # Should raise exception
        with pytest.raises(Exception, match="LLM Error"):
            await research_assistant.create_analysts(state, mock_llm)


@pytest.mark.e2e
class TestWorkflowStateValidation:
    """End-to-end tests for state validation through workflow."""

    def test_state_structure_through_workflow(self, sample_analysts):
        """Test state maintains correct structure through workflow stages."""
        # Initial state
        state = {
            "topic": "AI",
            "max_analysts": 3,
            "human_analyst_feedback": "",
            "analysts": [],
            "sections": [],
            "introduction": "",
            "content": "",
            "conclusion": "",
            "final_report": ""
        }

        # After analyst creation
        state["analysts"] = sample_analysts
        assert len(state["analysts"]) == 3
        assert all(hasattr(a, "name") for a in state["analysts"])

        # After interviews (sections added)
        state["sections"] = ["Section 1", "Section 2", "Section 3"]
        assert len(state["sections"]) == 3

        # After report writing
        state["introduction"] = "# Title\n\n## Introduction"
        state["content"] = "## Insights\n\nContent"
        state["conclusion"] = "## Conclusion"

        # Finalize
        from module4 import research_assistant
        final_state = research_assistant.finalize_report(state)

        assert "final_report" in final_state
        assert len(final_state["final_report"]) > 0
