"""Pytest configuration and shared fixtures."""

import pytest
import os
import sys
from unittest.mock import AsyncMock, Mock, MagicMock
from typing import AsyncGenerator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mcp import GetPromptResult
from mcp.types import TextContent, PromptMessage


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock ChatOpenAI instance."""
    llm = Mock(spec=ChatOpenAI)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mock response"))
    llm.invoke = Mock(return_value=AIMessage(content="Mock response"))

    # Mock structured output
    structured_llm = Mock()
    structured_llm.ainvoke = AsyncMock()
    structured_llm.invoke = Mock()
    llm.with_structured_output = Mock(return_value=structured_llm)

    return llm


@pytest.fixture
def mock_tavily_search():
    """Mock TavilySearch instance."""
    search = Mock()
    search.invoke = Mock(return_value={
        "results": [
            {
                "url": "https://example.com/1",
                "content": "Mock search result 1"
            },
            {
                "url": "https://example.com/2",
                "content": "Mock search result 2"
            }
        ]
    })
    return search


# ============================================================================
# Mock MCP Client Fixtures
# ============================================================================

@pytest.fixture
def mock_mcp_client():
    """Mock MultiServerMCPClient instance."""
    client = AsyncMock()

    async def mock_get_prompt(name: str, arguments: dict = None):
        """Mock get_prompt method."""
        content = f"Mock prompt for {name}"
        if arguments:
            content += f" with args: {arguments}"

        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=content
                    )
                )
            ]
        )

    client.get_prompt = mock_get_prompt
    return client


@pytest.fixture
async def mock_mcp_context(mock_mcp_client, monkeypatch):
    """Mock MCP client context manager."""
    from mcp_server import mcp_client_utils

    # Mock the global client
    monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

    async def mock_context():
        yield mock_mcp_client

    # Mock the context manager
    original_context = mcp_client_utils.mcp_client_context
    monkeypatch.setattr(
        mcp_client_utils,
        "mcp_client_context",
        lambda: mock_context()
    )

    return mock_mcp_client


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def sample_analyst():
    """Sample Analyst instance for testing."""
    from module4.model import Analyst

    return Analyst(
        name="Dr. Jane Smith",
        role="AI Research Specialist",
        affiliation="Tech University",
        description="Expert in machine learning and neural networks"
    )


@pytest.fixture
def sample_analysts():
    """List of sample Analyst instances."""
    from module4.model import Analyst

    return [
        Analyst(
            name="Dr. Jane Smith",
            role="AI Research Specialist",
            affiliation="Tech University",
            description="Expert in machine learning"
        ),
        Analyst(
            name="Prof. John Doe",
            role="Data Science Lead",
            affiliation="Research Lab",
            description="Expert in data analysis"
        ),
        Analyst(
            name="Dr. Alice Brown",
            role="NLP Researcher",
            affiliation="AI Institute",
            description="Expert in natural language processing"
        )
    ]


@pytest.fixture
def sample_generate_analysts_state(sample_analysts):
    """Sample GenerateAnalystsState for testing."""
    return {
        "topic": "Artificial Intelligence in Healthcare",
        "max_analysts": 3,
        "human_analyst_feedback": "",
        "analysts": sample_analysts
    }


@pytest.fixture
def sample_interview_state(sample_analyst):
    """Sample InterviewState for testing."""
    return {
        "messages": [
            HumanMessage(content="Hello, can you tell me about AI?")
        ],
        "max_num_turns": 2,
        "context": [],
        "analyst": sample_analyst,
        "interview": "",
        "sections": []
    }


@pytest.fixture
def sample_research_graph_state(sample_analysts):
    """Sample ResearchGraphState for testing."""
    return {
        "topic": "Artificial Intelligence in Healthcare",
        "max_analysts": 3,
        "human_analyst_feedback": "",
        "analysts": sample_analysts,
        "sections": [],
        "introduction": "",
        "content": "",
        "conclusion": "",
        "final_report": ""
    }


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key-123")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key-123")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


# ============================================================================
# Helper Functions
# ============================================================================

def create_mock_message(content: str, role: str = "assistant") -> AIMessage:
    """Helper to create mock messages."""
    if role == "assistant":
        return AIMessage(content=content)
    elif role == "human":
        return HumanMessage(content=content)
    else:
        return SystemMessage(content=content)
