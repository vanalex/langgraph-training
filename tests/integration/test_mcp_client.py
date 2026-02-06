"""Integration tests for MCP client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.integration
@pytest.mark.mcp
@pytest.mark.asyncio
class TestMCPClientIntegration:
    """Integration tests for MCP client utilities."""

    async def test_mcp_client_context_manager(self, monkeypatch):
        """Test MCP client context manager lifecycle."""
        from mcp_server import mcp_client_utils
        from fastmcp import MultiServerMCPClient

        # Mock the MultiServerMCPClient
        mock_client = AsyncMock(spec=MultiServerMCPClient)

        # Mock __aenter__ and __aexit__
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('mcp_server.mcp_client_utils.MultiServerMCPClient', return_value=mock_client):
            async with mcp_client_utils.mcp_client_context() as client:
                assert client is not None
                assert mcp_client_utils._CURRENT_CLIENT == client

            # After exiting context, should be None
            assert mcp_client_utils._CURRENT_CLIENT is None

    async def test_get_current_session_with_active_client(self, mock_mcp_client, monkeypatch):
        """Test get_current_session returns active client."""
        from mcp_server import mcp_client_utils

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        client = mcp_client_utils.get_current_session()

        assert client == mock_mcp_client

    def test_get_current_session_without_active_client(self, monkeypatch):
        """Test get_current_session raises error when no active client."""
        from mcp_server import mcp_client_utils

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", None)

        with pytest.raises(RuntimeError, match="No active MCP client"):
            mcp_client_utils.get_current_session()

    async def test_mcp_client_configuration(self, monkeypatch):
        """Test MCP client is configured with correct parameters."""
        from mcp_server import mcp_client_utils

        # Set custom URL
        test_url = "http://testserver:9000/mcp"
        monkeypatch.setenv("MCP_SERVER_URL", test_url)

        # Need to reload module to pick up env var
        import importlib
        importlib.reload(mcp_client_utils)

        # Verify URL was updated
        assert mcp_client_utils.MCP_SERVER_URL == test_url

    async def test_get_mcp_client_deprecated(self, monkeypatch):
        """Test deprecated get_mcp_client function."""
        from mcp_server import mcp_client_utils
        from fastmcp import MultiServerMCPClient

        mock_client = AsyncMock(spec=MultiServerMCPClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('mcp_server.mcp_client_utils.MultiServerMCPClient', return_value=mock_client):
            async with mcp_client_utils.get_mcp_client() as client:
                assert client is not None


@pytest.mark.integration
@pytest.mark.mcp
@pytest.mark.asyncio
@pytest.mark.slow
class TestMCPPromptRetrieval:
    """Integration tests for prompt retrieval through MCP."""

    async def test_multiple_prompt_retrievals(self, mock_mcp_client, monkeypatch):
        """Test retrieving multiple prompts in sequence."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Test multiple sequential calls
        analyst_prompt = await research_assistant.get_analyst_instructions("AI", "", 3)
        question_prompt = await research_assistant.get_question_instructions("Test goals")
        search_prompt = await research_assistant.get_search_instructions()

        assert analyst_prompt is not None
        assert question_prompt is not None
        assert search_prompt is not None

        # Verify MCP client was called multiple times
        assert mock_mcp_client.get_prompt.call_count >= 3

    async def test_prompt_with_arguments(self, mock_mcp_client, monkeypatch):
        """Test prompt retrieval with various argument types."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", mock_mcp_client)

        # Test with string arguments
        result = await research_assistant.get_answer_instructions(
            goals="Research AI",
            context="Context about AI"
        )

        assert result is not None

        # Verify arguments were passed correctly
        call_args = mock_mcp_client.get_prompt.call_args
        assert call_args is not None

    async def test_prompt_error_handling(self, monkeypatch):
        """Test error handling when MCP prompt retrieval fails."""
        from mcp_server import mcp_client_utils
        from module4 import research_assistant

        # Create a client that raises an error
        error_client = AsyncMock()
        error_client.get_prompt = AsyncMock(side_effect=Exception("MCP Error"))

        monkeypatch.setattr(mcp_client_utils, "_CURRENT_CLIENT", error_client)

        # Should raise the exception
        with pytest.raises(Exception, match="MCP Error"):
            await research_assistant.get_analyst_instructions("topic", "", 3)


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPServerConfiguration:
    """Tests for MCP server configuration."""

    def test_default_server_url(self):
        """Test default MCP server URL."""
        from mcp_server import mcp_client_utils

        # Should have a default URL
        assert mcp_client_utils.MCP_SERVER_URL is not None
        assert "http" in mcp_client_utils.MCP_SERVER_URL

    def test_custom_server_url(self, monkeypatch):
        """Test custom MCP server URL from environment."""
        custom_url = "http://custom-server:8080/mcp"
        monkeypatch.setenv("MCP_SERVER_URL", custom_url)

        # Reload module to pick up env var
        from mcp_server import mcp_client_utils
        import importlib
        importlib.reload(mcp_client_utils)

        assert mcp_client_utils.MCP_SERVER_URL == custom_url
