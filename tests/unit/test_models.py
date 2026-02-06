"""Unit tests for model classes."""

import pytest
from module4.model import (
    Analyst,
    Perspectives,
    SearchQuery,
    GenerateAnalystsState,
    InterviewState,
    ResearchGraphState
)


@pytest.mark.unit
class TestAnalyst:
    """Tests for Analyst model."""

    def test_analyst_creation(self):
        """Test creating an Analyst instance."""
        analyst = Analyst(
            name="Dr. Jane Smith",
            role="AI Researcher",
            affiliation="Tech University",
            description="Expert in machine learning"
        )

        assert analyst.name == "Dr. Jane Smith"
        assert analyst.role == "AI Researcher"
        assert analyst.affiliation == "Tech University"
        assert analyst.description == "Expert in machine learning"

    def test_analyst_persona_property(self, sample_analyst):
        """Test the persona property returns formatted string."""
        persona = sample_analyst.persona

        assert "Dr. Jane Smith" in persona
        assert "AI Research Specialist" in persona
        assert "Tech University" in persona
        assert "Expert in machine learning and neural networks" in persona

    def test_analyst_validation(self):
        """Test that Analyst validates required fields."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Analyst(
                name="Test",
                role="Role"
                # Missing required fields
            )


@pytest.mark.unit
class TestPerspectives:
    """Tests for Perspectives model."""

    def test_perspectives_creation(self, sample_analysts):
        """Test creating a Perspectives instance."""
        perspectives = Perspectives(analysts=sample_analysts)

        assert len(perspectives.analysts) == 3
        assert all(isinstance(a, Analyst) for a in perspectives.analysts)

    def test_perspectives_empty_list(self):
        """Test creating Perspectives with empty list."""
        perspectives = Perspectives(analysts=[])
        assert perspectives.analysts == []


@pytest.mark.unit
class TestSearchQuery:
    """Tests for SearchQuery model."""

    def test_search_query_creation(self):
        """Test creating a SearchQuery instance."""
        query = SearchQuery(search_query="artificial intelligence healthcare")

        assert query.search_query == "artificial intelligence healthcare"

    def test_search_query_empty(self):
        """Test SearchQuery with empty string."""
        query = SearchQuery(search_query="")
        assert query.search_query == ""


@pytest.mark.unit
class TestGenerateAnalystsState:
    """Tests for GenerateAnalystsState."""

    def test_state_structure(self, sample_generate_analysts_state):
        """Test state has correct structure."""
        state = sample_generate_analysts_state

        assert "topic" in state
        assert "max_analysts" in state
        assert "human_analyst_feedback" in state
        assert "analysts" in state

        assert isinstance(state["topic"], str)
        assert isinstance(state["max_analysts"], int)
        assert isinstance(state["analysts"], list)

    def test_state_with_feedback(self):
        """Test state with human feedback."""
        state = {
            "topic": "Test Topic",
            "max_analysts": 2,
            "human_analyst_feedback": "Focus on technical aspects",
            "analysts": []
        }

        assert state["human_analyst_feedback"] == "Focus on technical aspects"


@pytest.mark.unit
class TestInterviewState:
    """Tests for InterviewState."""

    def test_interview_state_structure(self, sample_interview_state):
        """Test interview state has correct structure."""
        state = sample_interview_state

        assert "messages" in state
        assert "max_num_turns" in state
        assert "context" in state
        assert "analyst" in state
        assert "interview" in state
        assert "sections" in state

    def test_interview_state_with_context(self, sample_analyst):
        """Test interview state with context."""
        state = {
            "messages": [],
            "max_num_turns": 3,
            "context": ["Context 1", "Context 2"],
            "analyst": sample_analyst,
            "interview": "Interview transcript",
            "sections": ["Section 1"]
        }

        assert len(state["context"]) == 2
        assert len(state["sections"]) == 1


@pytest.mark.unit
class TestResearchGraphState:
    """Tests for ResearchGraphState."""

    def test_research_graph_state_structure(self, sample_research_graph_state):
        """Test research graph state has correct structure."""
        state = sample_research_graph_state

        assert "topic" in state
        assert "max_analysts" in state
        assert "human_analyst_feedback" in state
        assert "analysts" in state
        assert "sections" in state
        assert "introduction" in state
        assert "content" in state
        assert "conclusion" in state
        assert "final_report" in state

    def test_research_graph_state_progression(self, sample_analysts):
        """Test state progression through workflow."""
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

        # After analysts created
        state["analysts"] = sample_analysts
        assert len(state["analysts"]) == 3

        # After sections written
        state["sections"] = ["Section 1", "Section 2", "Section 3"]
        assert len(state["sections"]) == 3

        # After report written
        state["introduction"] = "# Introduction"
        state["content"] = "## Content"
        state["conclusion"] = "## Conclusion"
        state["final_report"] = "Complete report"

        assert state["final_report"] == "Complete report"
