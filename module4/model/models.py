"""Data models for the research assistant."""

from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
import operator
from typing import Annotated
from langgraph.graph import MessagesState


class Analyst(BaseModel):
    """Represents an AI analyst with specific expertise."""
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    """Collection of analyst perspectives."""
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")


class SearchQuery(BaseModel):
    """Search query for information retrieval."""
    search_query: str = Field(None, description="Search query for retrieval.")


class GenerateAnalystsState(TypedDict):
    """State for analyst generation."""
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


class InterviewState(MessagesState):
    """State for conducting interviews."""
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class ResearchGraphState(TypedDict):
    """State for the overall research process."""
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
