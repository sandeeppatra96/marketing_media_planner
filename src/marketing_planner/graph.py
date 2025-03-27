import json
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langmem import create_manage_memory_tool, create_search_memory_tool
from marketing_planner import prompts
from marketing_planner.configuration import Configuration
from marketing_planner.state import EnhancedState, InputState, OutputState
from marketing_planner.tools import scrape_website, search, analyze_website
from marketing_planner.utils import init_model
from marketing_planner.shared import memory_store  # Import from shared module


manage_memory_tool = create_manage_memory_tool(namespace="marketing_memories")
search_memory_tool = create_search_memory_tool(namespace="marketing_memories")

async def call_agent_model(
    state: EnhancedState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Call the primary Language Model (LLM) to decide on the next research action."""
    # Load configuration from the provided RunnableConfig
    configuration = Configuration.from_runnable_config(config)

    # Define the 'MarketingPlan' tool, which is the user-defined plan schema
    marketing_plan_tool = {
        "name": "MarketingPlan",
        "description": "Call this when you have gathered all the relevant info",
        "parameters": {
            "type": "object",
            "properties": {
                "business_overview": {
                    "type": "object",
                    "properties": {
                        "industry": {"type": "string"},
                        "products": {"type": "array", "items": {"type": "string"}},
                        "target_audience": {"type": "string"},
                        "existing_marketing": {"type": "string"}
                    }
                },
                "competitor_insights": {
                    "type": "array", 
                    "items": {
                        "type": "object",
                        "properties": {
                            "competitor_name": {"type": "string"},
                            "ad_platforms": {"type": "array", "items": {"type": "string"}},
                            "audience": {"type": "string"},
                            "budget_estimate": {"type": "string"}
                        }
                    }
                },
                "recommended_channels": {"type": "array", "items": {"type": "string"}},
                "budget_allocation": {"type": "object", "additionalProperties": {"type": "number"}},
                "suggested_ad_creatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "platform": {"type": "string"},
                            "ad_type": {"type": "string"},
                            "creative": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
    
    # Define the AskUserInput tool for interactive refinement
    ask_user_input_tool = {
        "name": "AskUserInput",
        "description": "Call this when you need specific information from the user to improve the marketing plan",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "A specific question to ask the user"},
                "context": {"type": "string", "description": "The context or reason for asking this question"},
                "options": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Optional list of suggested response options"
                }
            },
            "required": ["question"]
        }
    }

    # Check if the research progress should be included in the prompt
    research_status = ""
    if state.research_progress:
        progress_items = []
        for key, val in state.research_progress.items():
            status = "✓" if val else "..."
            progress_items.append(f"- {key.replace('_', ' ').title()}: {status}")
        research_status = "\n\nCurrent research progress:\n" + "\n".join(progress_items)

    # Include user preferences in the prompt if available
    user_prefs = ""
    if state.user_preferences:
        pref_items = []
        for key, val in state.user_preferences.items():
            pref_items.append(f"- {key}: {val}")
        user_prefs = "\n\nUser preferences:\n" + "\n".join(pref_items)

    p = prompts.MAIN_PROMPT.format(
        website=state.website_url or "not provided yet"
    )
    
    # Add research status and user preferences to the prompt
    if research_status or user_prefs:
        p += f"\n{research_status}{user_prefs}"
    
    # Add specific instructions to avoid memory tool loops
    p += "\n\nIMPORTANT: Use memory tools sparingly. Only create or update memories for truly important business insights or user preferences. Do not create redundant memories or call memory tools unnecessarily. Focus primarily on using Search, ScrapeWebsite, and AnalyzeWebsite tools to gather information for the marketing plan."

    if not state.messages:
        messages = [HumanMessage(content=p)]
    else:
        messages = state.messages

    # Initialize the raw model with the provided configuration and bind the tools
    raw_model = init_model(config)
    
    # Determine which tools to include
    # If we've made more than 3 memory tool calls in a row, temporarily exclude memory tools
    memory_tool_calls = 0
    for idx in range(min(5, len(state.messages))):
        if idx >= len(state.messages):
            break
            
        msg = state.messages[-(idx+1)]
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] in ["ManageMemory", "SearchMemory"]:
                    memory_tool_calls += 1
    
    tools = [scrape_website, search, analyze_website, marketing_plan_tool, ask_user_input_tool]
    
    # Only include memory tools if we haven't seen too many consecutive memory calls
    if memory_tool_calls < 3:
        tools.extend([manage_memory_tool, search_memory_tool])
    
    # Include all tools including the AskUserInput tool
    model = raw_model.bind_tools(tools, tool_choice="any")
    
    response = cast(AIMessage, await model.ainvoke(messages))

    # Initialize marketing_plan to None
    marketing_plan = None
    asking_user = False
    user_question = None
    
    # Check if the response has tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "MarketingPlan":
                marketing_plan = tool_call["args"]
                break
            elif tool_call["name"] == "AskUserInput":
                asking_user = True
                user_question = tool_call["args"]
                break
                
    if marketing_plan is not None:
        # The agent is submitting their answer
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "MarketingPlan")
        ]
    elif asking_user:
        # The agent is asking the user a question
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "AskUserInput")
        ]
        
    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:  # If LLM didn't respect the tool_choice
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )
    
    return {
        "messages": response_messages,
        "marketing_plan": marketing_plan,
        "awaiting_user_input": asking_user,
        "last_question_context": user_question["context"] if asking_user and "context" in user_question else None,
        "loop_step": state.loop_step + 1,
    }


class MarketingPlanIsSatisfactory(BaseModel):
    """Validate whether the current marketing plan is satisfactory and complete."""

    reason: List[str] = Field(
        description="First, provide reasoning for why this is either good or bad as a final marketing plan. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the marketing plan is satisfactory. If not, you will continue researching."
    )
    improvement_instructions: Optional[str] = Field(
        description="If the plan is not satisfactory, provide clear and specific instructions on what needs to be improved or added.",
        default=None,
    )


async def reflect(
    state: EnhancedState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Validate the quality of the data enrichment agent's output."""

    p = prompts.MAIN_PROMPT.format(
        website=state.website_url or "not provided yet"
    )
    
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect.__name__} expects the last message in the state to be an AI message with tool calls."
            f" Got: {type(last_message)}"
        )
    
    # Ensure the last message has tool_calls
    if not last_message.tool_calls:
        raise ValueError(
            f"{reflect.__name__} expects the last message to have tool_calls."
        )
        
    messages = [HumanMessage(content=p)] + state.messages[:-1]
    
    presumed_plan = state.marketing_plan
    
    # Include user preferences in the evaluation
    user_prefs_str = ""
    if state.user_preferences:
        user_prefs_str = "\n\nUser preferences:\n" + "\n".join([
            f"- {k}: {v}" for k, v in state.user_preferences.items()
        ])
    
    checker_prompt = """I am thinking of creating a marketing plan with the information below. 
Is this good? Give your reasoning as well. 
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.
{user_prefs}

{presumed_plan}"""
    
    p1 = checker_prompt.format(
        presumed_plan=json.dumps(presumed_plan or {}, indent=2),
        user_prefs=user_prefs_str
    )
    messages.append(HumanMessage(content=p1))
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(MarketingPlanIsSatisfactory)
    response = cast(MarketingPlanIsSatisfactory, await bound_model.ainvoke(messages))
    
    # Find the MarketingPlan tool call - must exist based on the route_after_agent function
    marketing_plan_tool_call = next(
        tc for tc in last_message.tool_calls if tc["name"] == "MarketingPlan"
    )
    
    # Update plan revision counter
    plan_revision_count = state.plan_revision_count + 1
    
    if response.is_satisfactory and presumed_plan:
        # Create a tool message that correctly responds to the tool call
        return {
            "marketing_plan": presumed_plan, 
            "plan_revision_count": plan_revision_count,
            "messages": [
                ToolMessage(
                    tool_call_id=marketing_plan_tool_call["id"],  # Use the correct ID
                    content="\n".join(response.reason),
                    name="MarketingPlan",
                    status="success",
                )
            ],
        }
    else:
        # Create a tool message that correctly responds to the tool call
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=marketing_plan_tool_call["id"],  # Use the correct ID
                    content=f"Unsatisfactory response:\n{response.improvement_instructions}",
                    name="MarketingPlan",
                    status="error",
                )
            ],
            "plan_revision_count": plan_revision_count
        }


async def process_user_input(
    state: EnhancedState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Process user responses to questions from the agent."""
    
    # Identify the question that was asked
    second_last_message = state.messages[-2] if len(state.messages) >= 2 else None
    
    # Make sure we were awaiting user input
    if not state.awaiting_user_input or not second_last_message or not second_last_message.tool_calls:
        return {"awaiting_user_input": False}
    
    # Get the question that was asked
    tool_call = next((tc for tc in second_last_message.tool_calls if tc["name"] == "AskUserInput"), None)
    if not tool_call:
        return {"awaiting_user_input": False}
    
    # Get the user's response (last message)
    last_message = state.messages[-1]
    user_response = last_message.content if isinstance(last_message, HumanMessage) else ""
    
    # Update user preferences based on the question and response
    new_user_preferences = state.user_preferences.copy()
    
    # Try to match the question context to a preference key
    question = tool_call["args"].get("question", "")
    context = tool_call["args"].get("context", "")
    
    # Determine what type of question was asked and update preferences accordingly
    if "budget" in question.lower():
        new_user_preferences["budget"] = user_response
        new_research_progress = state.research_progress.copy()
        new_research_progress["budget_confirmed"] = True
        return {
            "user_preferences": new_user_preferences,
            "research_progress": new_research_progress,
            "awaiting_user_input": False
        }
    elif "timeline" in question.lower() or "launch" in question.lower():
        new_user_preferences["timeline"] = user_response
        new_research_progress = state.research_progress.copy()
        new_research_progress["timeline_confirmed"] = True
        return {
            "user_preferences": new_user_preferences,
            "research_progress": new_research_progress,
            "awaiting_user_input": False
        }
    elif "channel" in question.lower() or "platform" in question.lower():
        new_user_preferences["preferred_channels"] = user_response
        return {
            "user_preferences": new_user_preferences,
            "awaiting_user_input": False
        }
    elif "goal" in question.lower() or "objective" in question.lower():
        new_user_preferences["marketing_goals"] = user_response
        return {
            "user_preferences": new_user_preferences,
            "awaiting_user_input": False
        }
    elif "industry" in question.lower() or "business" in question.lower():
        # Verify business information
        new_research_progress = state.research_progress.copy()
        new_research_progress["website_analyzed"] = True
        return {
            "research_progress": new_research_progress,
            "awaiting_user_input": False
        }
    else:
        # Generic question - just store the response as a preference
        key = context.lower().replace(" ", "_") if context else "user_input"
        new_user_preferences[key] = user_response
        return {
            "user_preferences": new_user_preferences,
            "awaiting_user_input": False
        }



def route_after_agent(
    state: EnhancedState,
) -> Literal["reflect", "tools", "process_user_input", "call_agent_model", "__end__"]:
    """Schedule the next node after the agent's action."""
    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage):
        # If the last message is from the user and we were awaiting user input,
        # route to process_user_input
        if state.awaiting_user_input:
            return "process_user_input"
        return "call_agent_model"

    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        
        # Check for memory tool loop
        memory_tool_calls = 0
        for idx in range(min(5, len(state.messages))):
            if idx >= len(state.messages):
                break
                
            msg = state.messages[-(idx+1)]
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] in ["ManageMemory", "SearchMemory"]:
                        memory_tool_calls += 1
        
        # If we've seen too many memory tool calls in a row, 
        # force the agent back to the main path
        if memory_tool_calls >= 3 and tool_name in ["ManageMemory", "SearchMemory"]:
            return "call_agent_model"
            
        if tool_name == "MarketingPlan":
            return "reflect"
        elif tool_name == "AskUserInput":
            # The agent is asking the user a question
            # We don't route to process_user_input yet because we need to wait for the user's response
            return "__end__"  # Pause execution and wait for user input
        else:
            return "tools"
    else:
        return "tools"

def route_after_user_input(
    state: EnhancedState,
) -> Literal["call_agent_model"]:
    """Schedule the next node after processing user input."""
    # Always go back to the agent model after processing user input
    return "call_agent_model"


def route_after_checker(
    state: EnhancedState, config: RunnableConfig
) -> Literal["__end__", "call_agent_model"]:
    """Schedule the next node after the checker's evaluation."""
    configurable = Configuration.from_runnable_config(config)
    last_message = state.messages[-1]

    if state.loop_step < configurable.max_loops:
        if not state.marketing_plan: 
            return "call_agent_model"
        if not isinstance(last_message, ToolMessage):
            raise ValueError(
                f"{route_after_checker.__name__} expected a tool messages. Received: {type(last_message)}."
            )
        if last_message.status == "error":
            # Research deemed unsatisfactory
            return "call_agent_model"
        # It's great!
        return "__end__"
    else:
        return "__end__"


# Create the enhanced graph using EnhancedState
workflow = StateGraph(
    EnhancedState, input=InputState, output=OutputState, config_schema=Configuration
)

# Add all nodes to the graph
workflow.add_node(call_agent_model)
workflow.add_node(reflect)
workflow.add_node(process_user_input)
workflow.add_node("tools", ToolNode([
    search, 
    scrape_website, 
    analyze_website, 
    manage_memory_tool,
    search_memory_tool
]))

# Define the edges
workflow.add_edge("__start__", "call_agent_model")
workflow.add_conditional_edges("call_agent_model", route_after_agent)
workflow.add_edge("tools", "call_agent_model")
workflow.add_edge("process_user_input", "call_agent_model")
workflow.add_conditional_edges("reflect", route_after_checker)

# Update the compile to include the store
graph = workflow.compile(store=memory_store)
graph.name = "MarketingMediaPlanner"