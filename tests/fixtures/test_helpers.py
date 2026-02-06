"""Test helper utilities and mock factories."""

from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mcp import GetPromptResult
from mcp.types import TextContent, PromptMessage


class MockLLMFactory:
    """Factory for creating mock LLM instances."""

    @staticmethod
    def create_mock_llm(responses: List[str] = None) -> Mock:
        """Create a mock LLM with predefined responses.

        Args:
            responses: List of response strings. If None, returns generic responses.

        Returns:
            Mock LLM instance
        """
        llm = Mock()

        if responses:
            llm.ainvoke = AsyncMock(side_effect=[
                AIMessage(content=resp) for resp in responses
            ])
            llm.invoke = Mock(side_effect=[
                AIMessage(content=resp) for resp in responses
            ])
        else:
            llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mock response"))
            llm.invoke = Mock(return_value=AIMessage(content="Mock response"))

        # Mock structured output
        structured_llm = Mock()
        structured_llm.ainvoke = AsyncMock()
        structured_llm.invoke = Mock()
        llm.with_structured_output = Mock(return_value=structured_llm)

        return llm

    @staticmethod
    def create_structured_llm_mock(return_value: Any) -> Mock:
        """Create a mock for structured output LLM.

        Args:
            return_value: The structured object to return

        Returns:
            Mock structured LLM
        """
        structured_llm = Mock()
        structured_llm.ainvoke = AsyncMock(return_value=return_value)
        structured_llm.invoke = Mock(return_value=return_value)
        return structured_llm


class MockMCPFactory:
    """Factory for creating mock MCP client instances."""

    @staticmethod
    def create_mock_mcp_client(prompt_responses: Dict[str, str] = None) -> AsyncMock:
        """Create a mock MCP client with predefined prompt responses.

        Args:
            prompt_responses: Dict mapping prompt names to response strings

        Returns:
            Mock MCP client
        """
        client = AsyncMock()

        async def mock_get_prompt(name: str, arguments: dict = None):
            if prompt_responses and name in prompt_responses:
                content = prompt_responses[name]
            else:
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


class StateBuilder:
    """Builder for creating test state objects."""

    @staticmethod
    def create_generate_analysts_state(
        topic: str = "Test Topic",
        max_analysts: int = 3,
        feedback: str = "",
        analysts: List = None
    ) -> Dict:
        """Create a GenerateAnalystsState for testing."""
        return {
            "topic": topic,
            "max_analysts": max_analysts,
            "human_analyst_feedback": feedback,
            "analysts": analysts or []
        }

    @staticmethod
    def create_interview_state(
        analyst=None,
        messages: List = None,
        max_num_turns: int = 2,
        context: List = None
    ) -> Dict:
        """Create an InterviewState for testing."""
        if messages is None:
            messages = [HumanMessage(content="Hello")]

        return {
            "messages": messages,
            "max_num_turns": max_num_turns,
            "context": context or [],
            "analyst": analyst,
            "interview": "",
            "sections": []
        }

    @staticmethod
    def create_research_graph_state(
        topic: str = "Test Topic",
        max_analysts: int = 3,
        analysts: List = None,
        sections: List = None
    ) -> Dict:
        """Create a ResearchGraphState for testing."""
        return {
            "topic": topic,
            "max_analysts": max_analysts,
            "human_analyst_feedback": "",
            "analysts": analysts or [],
            "sections": sections or [],
            "introduction": "",
            "content": "",
            "conclusion": "",
            "final_report": ""
        }


class AnalystFactory:
    """Factory for creating test Analyst instances."""

    @staticmethod
    def create_analyst(
        name: str = "Dr. Test",
        role: str = "Test Role",
        affiliation: str = "Test Org",
        description: str = "Test description"
    ):
        """Create a single Analyst for testing."""
        from module4.model import Analyst
        return Analyst(
            name=name,
            role=role,
            affiliation=affiliation,
            description=description
        )

    @staticmethod
    def create_analysts(count: int = 3) -> List:
        """Create multiple Analyst instances for testing."""
        from module4.model import Analyst

        analysts = []
        for i in range(count):
            analysts.append(Analyst(
                name=f"Dr. Analyst {i+1}",
                role=f"Role {i+1}",
                affiliation=f"Organization {i+1}",
                description=f"Expert in area {i+1}"
            ))
        return analysts


class MessageFactory:
    """Factory for creating message instances."""

    @staticmethod
    def create_human_message(content: str) -> HumanMessage:
        """Create a HumanMessage."""
        return HumanMessage(content=content)

    @staticmethod
    def create_ai_message(content: str, name: str = None) -> AIMessage:
        """Create an AIMessage."""
        msg = AIMessage(content=content)
        if name:
            msg.name = name
        return msg

    @staticmethod
    def create_system_message(content: str) -> SystemMessage:
        """Create a SystemMessage."""
        return SystemMessage(content=content)

    @staticmethod
    def create_conversation(turns: int = 3) -> List:
        """Create a mock conversation with multiple turns."""
        messages = []
        for i in range(turns):
            messages.append(HumanMessage(content=f"Question {i+1}"))
            messages.append(AIMessage(content=f"Answer {i+1}"))
        return messages


def assert_state_structure(state: Dict, expected_keys: List[str]):
    """Assert that state has expected keys.

    Args:
        state: State dictionary to check
        expected_keys: List of keys that should be present
    """
    for key in expected_keys:
        assert key in state, f"State missing key: {key}"


def assert_message_sequence(messages: List, expected_types: List[type]):
    """Assert that messages follow expected type sequence.

    Args:
        messages: List of message objects
        expected_types: List of expected message types
    """
    assert len(messages) == len(expected_types), \
        f"Expected {len(expected_types)} messages, got {len(messages)}"

    for msg, expected_type in zip(messages, expected_types):
        assert isinstance(msg, expected_type), \
            f"Expected {expected_type.__name__}, got {type(msg).__name__}"
