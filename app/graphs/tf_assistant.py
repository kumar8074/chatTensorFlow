# ===================================================================================
# Project: ChatTensorFlow
# File: app/graphs/router.py
# Description: This file contains the implementation of main TensorFlowAssitant Graph.
#              It also uses the Researcher sub-Graph.
# Author: LALAN KUMAR
# Created: [15-05-2025]
# Updated: [09-06-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import asyncio
import os
import sys
from langgraph.graph import StateGraph, START, END
from typing import Any
from typing_extensions import TypedDict, cast, Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from app.graphs.states import AgentState, InputState, Router
from app.graphs.prompts import (
    ROUTER_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    MORE_INFO_SYSTEM_PROMPT,
    RESEARCH_PLAN_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_PROMPT
)
from app.core.llm import get_llm
from app.graphs.researcher import create_researcher_graph
from app.core.utils import format_docs

# QueryAnalyzer
async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Analyze the user's query and determine the appropriate routing"""
    llm = get_llm(
        streaming=config.get("streaming", False),
        callbacks=config.get("callbacks", [])
    )
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    #response = cast(Router, await llm.with_structured_output({"logic": str, "type": str}).ainvoke(messages))
    model= llm.with_structured_output(Router) # Important for Gemini or Antropic models
    response= await model.ainvoke(messages, config=config)
    return {"router": response}

# QueryRouter
def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next steps based on query classification"""
    _type = state.router["type"]
    if _type == "tensorflow":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type: {_type}")

# ask_for_more_info
async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information"""
    llm = get_llm(
        streaming=config.get("streaming", False), 
        callbacks=config.get("callbacks", [])
    )
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(logic=state.router["logic"])
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await llm.ainvoke(messages, config=config)
    return {"messages": [response]}

# Respond to general query
async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to tensorflow."""
    llm = get_llm(
        streaming=config.get("streaming", False), 
        callbacks=config.get("callbacks", [])
    )
    system_prompt = GENERAL_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await llm.ainvoke(messages, config=config)
    return {"messages": [response]}

# Create research plan
async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a tensorflow related query."""

    class Plan(TypedDict):
        """Generate research plan."""
        steps: list[str]

    llm = get_llm(
        streaming=config.get("streaming", False), 
        callbacks=config.get("callbacks", [])
    )
    model = llm.with_structured_output(Plan)
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + state.messages
    response = cast(Plan, await model.ainvoke(messages, config=config))
    return {"steps": response["steps"], "documents": "delete"}

# Conduct Research
async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan."""
    researcher_graph = create_researcher_graph()
    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    
    return {"documents": result["documents"], "steps": state.steps[1:]}

def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed."""
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"
    

# Conversation summarization node
async def summarize_conversation(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Incrementally summarize conversation history and trim old messages."""
    llm = get_llm(
        streaming=config.get("streaming", False),
        callbacks=config.get("callbacks", [])
    )
    
    # Determine new messages since last summary
    last_index = state.last_summarized_index or -1
    new_messages = state.messages[last_index + 1 :]
    
    # If no new messages, no update needed
    if not new_messages:
        return {}
    
    # Compose prompt to extend existing summary with new messages
    if state.summary:
        prompt = (
            f"This is the existing summary of the conversation:\n{state.summary}\n\n"
            "Extend the summary by incorporating the following new messages:\n"
        )
    else:
        prompt = "Create a summary of the following conversation messages:\n"
        
    def get_role(message):
        if isinstance(message, HumanMessage):
            return "Human"
        elif isinstance(message, AIMessage):
            return "Assistant"
        elif isinstance(message, SystemMessage):
            return "System"
        else:
            return "Unknown"
    
    # Format new messages as text for summarization
    new_lines = "\n".join(f"{get_role(m)}: {m.content}" for m in new_messages)
    prompt += new_lines + "\n\nNew summary:"
    
    # Use SystemMessage + HumanMessage for Gemini compatibility
    messages = [
        SystemMessage(content="You are an assistant that summarizes the conversation so far. Create a concise summary capturing the key points."),
        HumanMessage(content=prompt)
    ]
    
    # Call LLM to get updated summary
    summary_response = await llm.ainvoke(messages, config=config)
    updated_summary = summary_response.content.strip()
    
    # Prepare new messages list: summary system message + recent messages after new_messages
    # Keep last 2 messages after new_messages for context
    num_recent = 2
    recent_messages = state.messages[last_index + 1 + len(new_messages) :][-num_recent:]
    new_messages_list = [SystemMessage(content=f"Summary of conversation so far: {updated_summary}")] + recent_messages
    
    # Prepare RemoveMessage objects to delete all messages up to last summarized index + new_messages
    messages_to_remove = state.messages[: last_index + 1 + len(new_messages)]
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]
    
    return {
        "summary": updated_summary,
        "last_summarized_index": last_index + len(new_messages),
        "messages": new_messages_list + delete_messages,
    }

# Conditional function to decide whether to summarize or continue
def check_summarize(state: AgentState) -> Literal["summarize_conversation", "continue"]:
    # Summarize if token count exceeds threshold and there are new messages since last summary
    token_count = count_tokens_approximately(state.messages)
    last_index = state.last_summarized_index or -1
    if token_count >= 1000 and len(state.messages) > last_index + 1:
        return "summarize_conversation"
    return "continue"

async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research."""

    llm = get_llm(
        streaming=config.get("streaming", False), 
        callbacks=config.get("callbacks", [])
    )
    context = format_docs(state.documents)
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": prompt + "\n\nIMPORTANT: Always preserve code blocks with ```python and ``` markers. Never modify code content."}
    ] + state.messages
    response = await llm.ainvoke(messages, config=config)
    return {"messages": [response]}

def create_assistant_graph():
    """Create and return the main agent graph."""
    builder = StateGraph(AgentState, input=InputState)

    builder.add_node("analyze_and_route_query", analyze_and_route_query)
    builder.add_node("ask_for_more_info", ask_for_more_info)
    builder.add_node("respond_to_general_query", respond_to_general_query)
    builder.add_node("conduct_research", conduct_research)
    builder.add_node("create_research_plan", create_research_plan)
    builder.add_node("respond", respond)
    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_node("continue", lambda state, config: {})  # No-op dummy node
    
    # Entry point and other edges as before
    builder.add_edge(START, "analyze_and_route_query")
    builder.add_conditional_edges("analyze_and_route_query", route_query)
    builder.add_edge("create_research_plan", "conduct_research")
    builder.add_conditional_edges("conduct_research", check_finished)
    builder.add_edge("respond", "summarize_conversation")
    builder.add_conditional_edges(
        "summarize_conversation",
        check_summarize,
        path_map={
            "summarize_conversation": "continue",
            "continue": "continue",
        },
    )
    builder.add_edge("continue", END)
    builder.add_edge("ask_for_more_info", "create_research_plan")
    builder.add_edge("respond_to_general_query", END)

    
    # Instantiate MemorySaver checkpointer
    memory=MemorySaver()

    # Compile into a graph object
    graph = builder.compile(checkpointer=memory)
    graph.name = "TensorFlowAssistantGraph"
    
    return graph


# Example usage:
#graph=create_assistant_graph()
#print("Graph compiled successfully.")
#print(graph.nodes)

#input_state = AgentState(
    #messages=[HumanMessage(content="How to build convolutional neural network?")]
#)
# Optionally, define a config with thread_id and user_id for persistence
#config = {"configurable": {"thread_id": "1", "user_id": "user123"}}

#result = asyncio.run(graph.ainvoke(input_state,config))
#print(result)

#print("------------------------------------------------------------------------------------------",sep="\n")

#final_response = result["messages"][-1].content
#print(final_response)
