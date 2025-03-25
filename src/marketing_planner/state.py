"""Enhanced state definitions with user interaction tracking.

This module extends the basic State with additional fields to track
user preferences and interaction flow.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""
    website_url: Optional[str] = field(default=None)
    """The business website URL to analyze."""
    
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    """Messages track the primary execution state of the agent."""
    
    marketing_plan: Optional[Dict[str, Any]] = field(default=None)
    """The marketing plan data structure"""
    
    loop_step: Annotated[int, operator.add] = field(default=0)


@dataclass(kw_only=True)
class State(InputState):
    """A graph's State defines three main things.

    1. The structure of the data to be passed between nodes (which "channels" to read from/write to and their types)
    2. Default values for each field
    3. Reducers for the state's fields. Reducers are functions that determine how to apply updates to the state.
    See [Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) for more information.
    """

    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    """
    Messages track the primary execution state of the agent.
    """

    loop_step: Annotated[int, operator.add] = field(default=0)


@dataclass(kw_only=True)
class EnhancedState(State):
    """Enhanced state with additional fields for tracking user interactions."""
    
    awaiting_user_input: bool = field(default=False)
    """Flag to indicate when the system is waiting for user input."""
    
    plan_revision_count: Annotated[int, operator.add] = field(default=0)
    """Counter to track how many times the plan has been revised."""
    
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    """Store user preferences that have been collected during interaction."""
    
    last_question_context: Optional[str] = field(default=None)
    """Context about the last question asked to the user, for better follow-up."""
    
    research_progress: Dict[str, bool] = field(default_factory=lambda: {
        "website_analyzed": False,
        "competitors_analyzed": False,
        "channels_researched": False,
        "budget_confirmed": False,
        "timeline_confirmed": False
    })
    """Track research progress to guide the agent's next steps."""


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: Dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """