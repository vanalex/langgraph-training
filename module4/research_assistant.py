#!/usr/bin/env python3
"""
Research Assistant - A multi-agent system for automated research and report generation.

This application generates AI analyst personas, conducts parallel interviews with experts,
and synthesizes the findings into a comprehensive report.
"""

import os
import sys
import getpass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module4.model.models import GenerateAnalystsState, Perspectives, InterviewState, SearchQuery, ResearchGraphState

from mcp_server.mcp_client_utils import get_current_session, mcp_client_context

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

import requests

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Environment Setup
# ============================================================================

def _set_env(var: str):
    """Set environment variable if not already set."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def setup_environment():
    """Initialize environment variables."""
    _set_env("OPENAI_API_KEY")
    _set_env("TAVILY_API_KEY")
    _set_env("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "langchain-academy"


# ============================================================================
# Prompts - Now retrieved from MCP Server
# ============================================================================

async def get_analyst_instructions(topic: str, human_analyst_feedback: str, max_analysts: int) -> str:
    """Get analyst instructions from MCP server."""
    client = get_current_session()
    result = await client.get_prompt("analyst-instructions",
                     arguments={"topic": topic, "human_analyst_feedback": human_analyst_feedback, "max_analysts": str(max_analysts)})
    return result.messages[0].content.text

async def get_question_instructions(goals: str) -> str:
    """Get question instructions from MCP server."""
    client = get_current_session()
    result = await client.get_prompt("question-instructions", arguments={"goals": goals})
    return result.messages[0].content.text

async def get_search_instructions() -> SystemMessage:
    """Get search instructions from MCP server."""
    client = get_current_session()
    result = await client.get_prompt("search-instructions")
    return SystemMessage(content=result.messages[0].content.text)

async def get_answer_instructions(goals: str, context: str) -> str:
    """Get answer instructions from MCP server."""
    client = get_current_session()
    result = await client.get_prompt("answer-instructions", arguments={"goals": goals, "context": context})
    return result.messages[0].content.text

async def get_section_writer_instructions(focus: str) -> str:
    """Get section writer instructions from MCP server."""
    client = get_current_session()
    result = await client.get_prompt("section-writer-instructions", arguments={"focus": focus})
    return result.messages[0].content.text

async def get_report_writer_instructions(topic: str, context: str) -> str:
    """Get report writer instructions from MCP server."""
    client = get_current_session()
    result = await client.get_prompt("report-writer-instructions", arguments={"topic": topic, "context": context})
    return result.messages[0].content.text

async def get_intro_conclusion_instructions(topic: str, formatted_str_sections: str) -> str:
    """Get intro/conclusion instructions from MCP server."""
    client = get_current_session()
    result = await client.get_prompt("intro-conclusion-instructions",
                     arguments={"topic": topic, "formatted_str_sections": formatted_str_sections})
    return result.messages[0].content.text


# ============================================================================
# Analyst Generation Nodes
# ============================================================================

async def create_analysts(state: GenerateAnalystsState, llm):
    """Create analysts based on topic and feedback."""
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    structured_llm = llm.with_structured_output(Perspectives)
    system_message = await get_analyst_instructions(topic, human_analyst_feedback, max_analysts)

    analysts = await structured_llm.ainvoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of analysts.")
    ])

    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interrupted on."""
    pass


def should_continue(state: GenerateAnalystsState):
    """Return the next node to execute."""
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END


# ============================================================================
# Interview Nodes
# ============================================================================

async def generate_question(state: InterviewState, llm):
    """Generate a question for the expert."""
    analyst = state["analyst"]
    messages = state["messages"]

    system_message = await get_question_instructions(analyst.persona)
    question = await llm.ainvoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}


async def search_web(state: InterviewState, llm, tavily_search):
    """Retrieve docs from web search."""
    structured_llm = llm.with_structured_output(SearchQuery)
    search_instructions = await get_search_instructions()
    search_query = await structured_llm.ainvoke([search_instructions] + state['messages'])

    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data)

    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in search_docs
    ])

    return {"context": [formatted_search_docs]}


async def search_wikipedia(state: InterviewState, llm):
    """Retrieve docs from Wikipedia (no extra dependency)."""
    structured_llm = llm.with_structured_output(SearchQuery)
    search_instructions = await get_search_instructions()
    search_query = await structured_llm.ainvoke([search_instructions] + state['messages'])

    search_docs = _wikipedia_search(search_query.search_query, max_docs=2)

    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page=""/>\n{doc.page_content}\n</Document>'
        for doc in search_docs
    ])

    return {"context": [formatted_search_docs]}


async def generate_answer(state: InterviewState, llm):
    """Generate an answer to the analyst's question."""
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    system_message = await get_answer_instructions(analyst.persona, str(context))
    answer = await llm.ainvoke([SystemMessage(content=system_message)] + messages)
    answer.name = "expert"

    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """Save the interview transcript."""
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer."""
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    if num_responses >= max_num_turns:
        return 'save_interview'

    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'

    return "ask_question"


async def write_section(state: InterviewState, llm):
    """Write a section based on the interview."""
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    system_message = await get_section_writer_instructions(analyst.description)
    section = await llm.ainvoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write your section: {context}")
    ])

    return {"sections": [section.content]}


# ============================================================================
# Report Generation Nodes
# ============================================================================

def initiate_all_interviews(state: ResearchGraphState):
    """Map step to initiate all interviews in parallel."""
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"

    topic = state["topic"]
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
        })
        for analyst in state["analysts"]
    ]


async def write_report(state: ResearchGraphState, llm):
    """Write the main report content."""
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    system_message = await get_report_writer_instructions(topic, formatted_str_sections)
    report = await llm.ainvoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos.")
    ])

    return {"content": report.content}


async def write_introduction(state: ResearchGraphState, llm):
    """Write the report introduction."""
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    instructions = await get_intro_conclusion_instructions(topic, formatted_str_sections)
    intro = await llm.ainvoke([SystemMessage(content=instructions), HumanMessage(content="Write the report introduction")])

    return {"introduction": intro.content}


async def write_conclusion(state: ResearchGraphState, llm):
    """Write the report conclusion."""
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    instructions = await get_intro_conclusion_instructions(topic, formatted_str_sections)
    conclusion = await llm.ainvoke([SystemMessage(content=instructions), HumanMessage(content="Write the report conclusion")])

    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """Combine all sections into the final report."""
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


# ============================================================================
# Wikipedia (no extra dependency) helper
# ============================================================================

def _wikipedia_search(query: str, max_docs: int = 2) -> list[Document]:
    """
    Lightweight Wikipedia search using the MediaWiki API (no `wikipedia` PyPI dependency).
    Returns LangChain `Document` objects with `metadata["source"]` set to the page URL.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "langgraph-training/1.0 (research_assistant)"})

    search_resp = session.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_docs,
            "format": "json",
        },
        timeout=20,
    )
    search_resp.raise_for_status()
    hits = search_resp.json().get("query", {}).get("search", [])

    docs: list[Document] = []
    for hit in hits:
        pageid = hit.get("pageid")
        title = hit.get("title")
        if not pageid:
            continue

        page_resp = session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts|info",
                "pageids": pageid,
                "explaintext": 1,
                "exintro": 0,
                "inprop": "url",
                "format": "json",
            },
            timeout=20,
        )
        page_resp.raise_for_status()
        page = page_resp.json().get("query", {}).get("pages", {}).get(str(pageid), {})
        extract = (page.get("extract") or "").strip()
        fullurl = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        if extract:
            docs.append(
                Document(
                    page_content=extract,
                    metadata={"source": fullurl, "title": title, "pageid": pageid},
                )
            )

    return docs


# ============================================================================
# Analyst Generation Nodes
# ============================================================================

def build_interview_graph(llm, tavily_search):
    """Build the interview subgraph."""
    interview_builder = StateGraph(InterviewState)

    async def _ask_question(state):
        return await generate_question(state, llm)

    async def _search_web(state):
        return await search_web(state, llm, tavily_search)

    async def _search_wikipedia(state):
        return await search_wikipedia(state, llm)

    async def _answer_question(state):
        return await generate_answer(state, llm)

    async def _write_section(state):
        return await write_section(state, llm)

    interview_builder.add_node("ask_question", _ask_question)
    interview_builder.add_node("search_web", _search_web)
    interview_builder.add_node("search_wikipedia", _search_wikipedia)
    interview_builder.add_node("answer_question", _answer_question)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", _write_section)

    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    memory = MemorySaver()
    return interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")


def build_research_graph(llm, interview_graph):
    """Build the main research graph."""
    builder = StateGraph(ResearchGraphState)

    async def _create_analysts(state):
        return await create_analysts(state, llm)

    async def _write_report(state):
        return await write_report(state, llm)

    async def _write_introduction(state):
        return await write_introduction(state, llm)

    async def _write_conclusion(state):
        return await write_conclusion(state, llm)

    builder.add_node("create_analysts", _create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", _write_report)
    builder.add_node("write_introduction", _write_introduction)
    builder.add_node("write_conclusion", _write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    memory = MemorySaver()
    return builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)


# ============================================================================
# Main Application
# ============================================================================

async def run_research_assistant(topic: str, max_analysts: int = 3, thread_id: str = "1"):
    """Run the research assistant workflow."""
    # Setup
    setup_environment()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tavily_search = TavilySearch(max_results=3)

    # Build graphs
    interview_graph = build_interview_graph(llm, tavily_search)
    research_graph = build_research_graph(llm, interview_graph)

    thread = {"configurable": {"thread_id": thread_id}}

    # Generate analysts
    print("=" * 80)
    print("GENERATING ANALYSTS")
    print("=" * 80)
    async for event in research_graph.astream(
        {"topic": topic, "max_analysts": max_analysts},
        thread,
        stream_mode="values"
    ):
        analysts = event.get('analysts', '')
        if analysts:
            for analyst in analysts:
                print(f"\nName: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)

    # Get human feedback
    print("\n" + "=" * 80)
    print("ANALYST FEEDBACK")
    print("=" * 80)
    feedback = input("\nProvide feedback on analysts (or press Enter to continue): ").strip()

    if feedback:
        research_graph.update_state(
            thread,
            {"human_analyst_feedback": feedback},
            as_node="human_feedback"
        )

        # Show updated analysts
        async for event in research_graph.astream(None, thread, stream_mode="values"):
            analysts = event.get('analysts', '')
            if analysts:
                print("\nUpdated analysts:")
                for analyst in analysts:
                    print(f"\nName: {analyst.name}")
                    print(f"Affiliation: {analyst.affiliation}")
                    print(f"Role: {analyst.role}")
                    print(f"Description: {analyst.description}")
                    print("-" * 50)

        # Ask for confirmation
        confirm = input("\nProceed with these analysts? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Research cancelled.")
            return

    # Finalize analysts
    research_graph.update_state(
        thread,
        {"human_analyst_feedback": None},
        as_node="human_feedback"
    )

    # Conduct research
    print("\n" + "=" * 80)
    print("CONDUCTING RESEARCH")
    print("=" * 80)
    async for event in research_graph.astream(None, thread, stream_mode="updates"):
        node_name = next(iter(event.keys()))
        print(f"Processing: {node_name}")

    # Get final report
    final_state = await research_graph.aget_state(thread)
    report = final_state.values.get('final_report')

    # Display report
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(report)

    return report


def main():
    """Main entry point."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Research Assistant - Automated research and report generation")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--max-analysts", type=int, default=3, help="Maximum number of analysts (default: 3)")
    parser.add_argument("--thread-id", default="1", help="Thread ID for checkpointing (default: 1)")

    args = parser.parse_args()

    async def run_with_mcp():
        async with mcp_client_context():
            await run_research_assistant(args.topic, args.max_analysts, args.thread_id)

    asyncio.run(run_with_mcp())


if __name__ == "__main__":
    main()
