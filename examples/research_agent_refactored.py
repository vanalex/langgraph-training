"""Refactored Research Assistant using new architecture."""

from typing import List, Dict, Any
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from core.agents.base_agent import BaseAgent, AgentContext
from core.agents.mixins import MCPMixin, LLMMixin, SearchMixin
from core.config.settings import load_config
from core.graph.builder import GraphBuilder
from core.graph.edges import EdgeBuilder

from module4.model import (
    Analyst,
    Perspectives,
    SearchQuery,
    ResearchGraphState
)


class ResearchAgent(BaseAgent, MCPMixin, LLMMixin, SearchMixin):
    """Research assistant agent using new architecture.

    Combines multiple mixins for MCP, LLM, and search functionality.
    """

    def get_state_schema(self) -> type:
        """Get the state schema."""
        return ResearchGraphState

    def build_graph(self) -> Any:
        """Build the research graph."""
        builder = GraphBuilder(ResearchGraphState)

        # Add nodes
        builder.add_node("create_analysts", self._create_analysts_wrapper)
        builder.add_node("human_feedback", self._human_feedback)
        builder.add_node("conduct_interview", self._build_interview_subgraph())
        builder.add_node("write_report", self._write_report_wrapper)
        builder.add_node("write_introduction", self._write_introduction_wrapper)
        builder.add_node("write_conclusion", self._write_conclusion_wrapper)
        builder.add_node("finalize_report", self._finalize_report)

        # Add edges
        builder.add_edge("START", "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")

        # Conditional edge for feedback
        feedback_router = EdgeBuilder.create_feedback_router(
            "human_analyst_feedback",
            "create_analysts",
            "conduct_interview"
        )
        builder.add_conditional_edge("human_feedback", feedback_router,
                                    ["create_analysts", "conduct_interview"])

        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(
            ["write_conclusion", "write_report", "write_introduction"],
            "finalize_report"
        )
        builder.add_edge("finalize_report", "END")

        return builder.build()

    # Node wrappers
    async def _create_analysts_wrapper(self, state: Dict) -> Dict:
        """Wrapper for create_analysts node."""
        topic = state['topic']
        max_analysts = state['max_analysts']
        feedback = state.get('human_analyst_feedback', '')

        # Get prompt from MCP
        system_message = await self.get_prompt(
            "analyst-instructions",
            {"topic": topic, "human_analyst_feedback": feedback,
             "max_analysts": str(max_analysts)}
        )

        # Generate analysts using LLM
        perspectives = await self.generate_structured(
            [
                SystemMessage(content=system_message),
                HumanMessage(content="Generate the set of analysts.")
            ],
            schema=Perspectives
        )

        return {"analysts": perspectives.analysts}

    def _human_feedback(self, state: Dict) -> Dict:
        """No-op node for human feedback."""
        return {}

    async def _write_report_wrapper(self, state: Dict) -> Dict:
        """Wrapper for write_report node."""
        sections = state["sections"]
        topic = state["topic"]

        formatted_sections = "\n\n".join([f"{section}" for section in sections])
        system_message = await self.get_prompt(
            "report-writer-instructions",
            {"topic": topic, "context": formatted_sections}
        )

        report = await self.generate([
            SystemMessage(content=system_message),
            HumanMessage(content="Write a report based upon these memos.")
        ])

        return {"content": report.content}

    async def _write_introduction_wrapper(self, state: Dict) -> Dict:
        """Wrapper for write_introduction node."""
        sections = state["sections"]
        topic = state["topic"]

        formatted_sections = "\n\n".join([f"{section}" for section in sections])
        instructions = await self.get_prompt(
            "intro-conclusion-instructions",
            {"topic": topic, "formatted_str_sections": formatted_sections}
        )

        intro = await self.generate([
            SystemMessage(content=instructions),
            HumanMessage(content="Write the report introduction")
        ])

        return {"introduction": intro.content}

    async def _write_conclusion_wrapper(self, state: Dict) -> Dict:
        """Wrapper for write_conclusion node."""
        sections = state["sections"]
        topic = state["topic"]

        formatted_sections = "\n\n".join([f"{section}" for section in sections])
        instructions = await self.get_prompt(
            "intro-conclusion-instructions",
            {"topic": topic, "formatted_str_sections": formatted_sections}
        )

        conclusion = await self.generate([
            SystemMessage(content=instructions),
            HumanMessage(content="Write the report conclusion")
        ])

        return {"conclusion": conclusion.content}

    def _finalize_report(self, state: Dict) -> Dict:
        """Finalize the report by combining all sections."""
        content = state["content"]

        if content.startswith("## Insights"):
            content = content.strip("## Insights")

        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except:
                sources = None
        else:
            sources = None

        final_report = (
            state["introduction"] + "\n\n---\n\n" +
            content + "\n\n---\n\n" +
            state["conclusion"]
        )

        if sources is not None:
            final_report += "\n\n## Sources\n" + sources

        return {"final_report": final_report}

    def _build_interview_subgraph(self):
        """Build the interview subgraph."""
        # This would be built similarly using GraphBuilder
        # Placeholder for now
        from module4 import research_assistant
        llm = self.get_llm()
        search = self.initialize_search()
        return research_assistant.build_interview_graph(llm, search)


# Usage example
async def main():
    """Run the refactored research assistant."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "research_assistant.yaml"
    config = load_config(config_path)

    # Create agent context
    context = AgentContext(config=config)

    # Create agent
    agent = ResearchAgent(config, context)

    # Compile graph
    agent.compile_graph(interrupt_before=config.interrupt_before)

    # Run agent
    result = await agent.run({
        "topic": "Artificial Intelligence in Healthcare",
        "max_analysts": 3,
        "human_analyst_feedback": "",
        "analysts": [],
        "sections": [],
        "introduction": "",
        "content": "",
        "conclusion": "",
        "final_report": ""
    }, config={"configurable": {"thread_id": "1"}})

    print("Final Report:")
    print(result.get("final_report", "No report generated"))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
