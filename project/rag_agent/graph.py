import os
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from functools import partial

from .graph_state import State, AgentState
from .nodes import *
from .edges import *


def create_agent_graph(llm, tool_factory):
    checkpointer = InMemorySaver()

    local_tools = tool_factory.get_local_tools()
    local_tool_node = ToolNode(local_tools)

    print("Compiling agent graph...")

    # ---------------------------------------------------------
    # PART 1: Inner Agent Subgraph (per question)
    # ---------------------------------------------------------
    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("local_agent", partial(agent_node, llm=llm, tools=local_tools, system_prompt=get_local_rag_prompt()))
    agent_builder.add_node("local_tools", local_tool_node)
    agent_builder.add_node("check_local_result", partial(check_local_result_with_llm, llm=llm))
    agent_builder.add_node("prepare_retry", prepare_retry)
    agent_builder.add_node("extract_answer", extract_final_answer)

    def route_after_local_check(state: AgentState):
        target = state.get("fallback_route", "extract_answer")
        print(f"[ROUTE] Routing to: {target}")
        return target

    agent_builder.add_edge(START, "local_agent")
    agent_builder.add_conditional_edges(
        "local_agent",
        tools_condition,
        {"tools": "local_tools", END: "check_local_result"}
    )
    agent_builder.add_edge("local_tools", "local_agent")
    agent_builder.add_conditional_edges(
        "check_local_result",
        route_after_local_check,
        {"extract_answer": "extract_answer", "retry": "prepare_retry"}
    )
    agent_builder.add_edge("prepare_retry", "local_agent")
    agent_builder.add_edge("extract_answer", END)

    agent_processor = agent_builder.compile()

    # ---------------------------------------------------------
    # PART 2: Main Graph
    # ---------------------------------------------------------
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize", partial(analyze_chat_and_summarize, llm=llm))
    graph_builder.add_node("analyze_rewrite", partial(analyze_and_rewrite_query, llm=llm))
    graph_builder.add_node("human_input", human_input_node)
    graph_builder.add_node("process_question", agent_processor)
    graph_builder.add_node("aggregate", partial(aggregate_responses, llm=llm))

    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "analyze_rewrite")
    graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
    graph_builder.add_edge("human_input", "analyze_rewrite")
    graph_builder.add_edge("process_question", "aggregate")
    graph_builder.add_edge("aggregate", END)

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_input"]
    )

    # Save graph diagrams as Mermaid files with readable theme
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    def apply_theme(mermaid_code: str) -> str:
        import re
        mermaid_code = re.sub(
            r"classDef default fill:[^,\n]+",
            "classDef default fill:#2d6cdf,color:#ffffff,stroke:#1a4fa0",
            mermaid_code
        )
        return mermaid_code

    try:
        with open(os.path.join(repo_root, "agent_graph.mmd"), "w", encoding="utf-8") as f:
            f.write(apply_theme(agent_graph.get_graph(xray=True).draw_mermaid()))
        with open(os.path.join(repo_root, "agent_subgraph.mmd"), "w", encoding="utf-8") as f:
            f.write(apply_theme(agent_processor.get_graph(xray=True).draw_mermaid()))
        print("Graph diagrams saved: agent_graph.mmd, agent_subgraph.mmd")
    except Exception as e:
        print(f"Note: Could not save graph diagrams: {e}")

    print("Agent graph compiled successfully.")
    return agent_graph
