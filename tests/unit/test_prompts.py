"""Unit tests for prompt functions."""

import pytest
from unittest.mock import AsyncMock, patch
from mcp import GetPromptResult
from mcp.types import TextContent, PromptMessage

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.unit
@pytest.mark.asyncio
class TestPromptFunctions:
    """Tests for prompt retrieval functions."""

    @pytest.fixture
    def mock_prompt_result(self):
        """Create a mock GetPromptResult."""
        def create_result(content: str):
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
        return create_result

    async def test_get_analyst_instructions(self, mock_mcp_client, monkeypatch, mock_prompt_result):
        """Test get_analyst_instructions returns prompt text."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        # Mock the current session
        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Set up mock response
        expected_content = "Test analyst instructions"
        mock_mcp_client.get_prompt = AsyncMock(
            return_value=mock_prompt_result(expected_content)
        )

        result = await research_assistant.get_analyst_instructions(
            topic="AI",
            human_analyst_feedback="",
            max_analysts=3
        )

        assert result == expected_content
        mock_mcp_client.get_prompt.assert_called_once_with(
            "analyst-instructions",
            arguments={"topic": "AI", "human_analyst_feedback": "", "max_analysts": "3"}
        )

    async def test_get_question_instructions(self, mock_mcp_client, monkeypatch, mock_prompt_result):
        """Test get_question_instructions returns prompt text."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        expected_content = "Test question instructions"
        mock_mcp_client.get_prompt = AsyncMock(
            return_value=mock_prompt_result(expected_content)
        )

        result = await research_assistant.get_question_instructions(
            goals="Test goals"
        )

        assert result == expected_content
        mock_mcp_client.get_prompt.assert_called_once_with(
            "question-instructions",
            arguments={"goals": "Test goals"}
        )

    async def test_get_search_instructions(self, mock_mcp_client, monkeypatch, mock_prompt_result):
        """Test get_search_instructions returns SystemMessage."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant
        from langchain_core.messages import SystemMessage

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        expected_content = "Test search instructions"
        mock_mcp_client.get_prompt = AsyncMock(
            return_value=mock_prompt_result(expected_content)
        )

        result = await research_assistant.get_search_instructions()

        assert isinstance(result, SystemMessage)
        assert result.content == expected_content
        mock_mcp_client.get_prompt.assert_called_once_with("search-instructions")

    async def test_get_answer_instructions(self, mock_mcp_client, monkeypatch, mock_prompt_result):
        """Test get_answer_instructions with context."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        expected_content = "Test answer instructions"
        mock_mcp_client.get_prompt = AsyncMock(
            return_value=mock_prompt_result(expected_content)
        )

        result = await research_assistant.get_answer_instructions(
            goals="Test goals",
            context="Test context"
        )

        assert result == expected_content
        mock_mcp_client.get_prompt.assert_called_once_with(
            "answer-instructions",
            arguments={"goals": "Test goals", "context": "Test context"}
        )

    async def test_get_section_writer_instructions(self, mock_mcp_client, monkeypatch, mock_prompt_result):
        """Test get_section_writer_instructions."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        expected_content = "Test section writer instructions"
        mock_mcp_client.get_prompt = AsyncMock(
            return_value=mock_prompt_result(expected_content)
        )

        result = await research_assistant.get_section_writer_instructions(
            focus="Test focus"
        )

        assert result == expected_content
        mock_mcp_client.get_prompt.assert_called_once_with(
            "section-writer-instructions",
            arguments={"focus": "Test focus"}
        )

    async def test_get_report_writer_instructions(self, mock_mcp_client, monkeypatch, mock_prompt_result):
        """Test get_report_writer_instructions."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        expected_content = "Test report writer instructions"
        mock_mcp_client.get_prompt = AsyncMock(
            return_value=mock_prompt_result(expected_content)
        )

        result = await research_assistant.get_report_writer_instructions(
            topic="Test topic",
            context="Test context"
        )

        assert result == expected_content
        mock_mcp_client.get_prompt.assert_called_once_with(
            "report-writer-instructions",
            arguments={"topic": "Test topic", "context": "Test context"}
        )

    async def test_get_intro_conclusion_instructions(self, mock_mcp_client, monkeypatch, mock_prompt_result):
        """Test get_intro_conclusion_instructions."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        expected_content = "Test intro/conclusion instructions"
        mock_mcp_client.get_prompt = AsyncMock(
            return_value=mock_prompt_result(expected_content)
        )

        result = await research_assistant.get_intro_conclusion_instructions(
            topic="Test topic",
            formatted_str_sections="Section 1\nSection 2"
        )

        assert result == expected_content
        mock_mcp_client.get_prompt.assert_called_once_with(
            "intro-conclusion-instructions",
            arguments={"topic": "Test topic", "formatted_str_sections": "Section 1\nSection 2"}
        )

    async def test_prompt_function_error_handling(self, monkeypatch):
        """Test prompt functions handle missing MCP client gracefully."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        # Set no current client
        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", None)

        with pytest.raises(RuntimeError, match="No active MCP client"):
            await research_assistant.get_analyst_instructions("topic", "", 3)
