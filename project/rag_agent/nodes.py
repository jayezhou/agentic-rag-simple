import warnings
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
import config


async def analyze_chat_and_summarize(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    existing_summary = state.get("conversation_summary", "")

    conversation = "Recent conversation:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    if existing_summary.strip():
        summary_prompt = f"""Existing summary:\n{existing_summary}\n\nNew conversation:\n{conversation}\n\nTask: Update the summary by ADDING new information. Keep all existing entity information and append new facts. Each entity should be listed separately."""
    else:
        summary_prompt = conversation

    summary_response = await llm.with_config(temperature=0.2).ainvoke(
        [SystemMessage(content=get_conversation_summary_prompt())] + [HumanMessage(content=summary_prompt)]
    )

    print("\n" + "="*80)
    print("[CONVERSATION_SUMMARY] Generated Summary:")
    print(summary_response.content)
    print("="*80 + "\n")

    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}


async def analyze_and_rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    print("\n" + "="*80)
    print("[ANALYZE_REWRITE] Input:")
    print(f"User Query: {last_message.content}")
    print(f"Conversation Summary: {conversation_summary if conversation_summary else '(empty)'}")
    print("="*80 + "\n")

    context_section = (
        f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else ""
    ) + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0).with_structured_output(QueryAnalysis)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
            response = await llm_with_structure.ainvoke(
                [SystemMessage(content=get_query_analysis_prompt())] + [HumanMessage(content=context_section)]
            )
    except Exception as e:
        print(f"Warning: Structured output parsing failed: {e}")
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content="抱歉，我无法理解您的问题，请您重新描述一下。")]
        }

    if len(response.questions) > 0 and response.is_clear:
        delete_all = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if not isinstance(m, SystemMessage)
        ]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions
        }
    else:
        clarification = (
            response.clarification_needed
            if (response.clarification_needed and len(response.clarification_needed.strip()) > 10)
            else "I need more information to understand your question."
        )
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)]
        }


def human_input_node(state: State):
    return {}


async def agent_node(state: AgentState, llm, tools, system_prompt):
    llm_with_tools = llm.bind_tools(tools)
    sys_msg = SystemMessage(content=system_prompt)

    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        response = await llm_with_tools.ainvoke([sys_msg] + [human_msg])
        return {"messages": [human_msg, response]}

    return {"messages": [await llm_with_tools.ainvoke([sys_msg] + state["messages"])]}


def extract_final_answer(state: AgentState):
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return {
                "final_answer": msg.content,
                "agent_answers": [{
                    "index": state["question_index"],
                    "question": state["question"],
                    "answer": msg.content,
                }]
            }
    return {
        "final_answer": "Unable to generate an answer.",
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": "Unable to generate an answer.",
        }]
    }


async def aggregate_responses(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += f"\nAnswer {i}:\n{ans['answer']}\n"

    user_message = HumanMessage(
        content=f"Original user question: {state['originalQuery']}\nRetrieved answers:{formatted_answers}"
    )
    synthesis_response = await llm.ainvoke(
        [SystemMessage(content=get_aggregation_prompt())] + [user_message]
    )

    return {"messages": [AIMessage(content=synthesis_response.content)]}


async def check_local_result_with_llm(state: AgentState, llm):
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            last_message = msg.content
            break

    if not last_message:
        return {"fallback_route": "extract_answer"}

    check_prompt = get_check_local_result_prompt(state['question'], last_message)

    try:
        response = await llm.with_config(temperature=0.1).ainvoke([HumanMessage(content=check_prompt)])
        decision = response.content.strip().upper()

        print(f"\n[CHECK_LOCAL_RESULT] LLM Decision: {decision}")
        print(f"Question: {state['question']}")
        print(f"Local Result Preview: {last_message[:200]}...\n")

        if decision == "LOCAL":
            target_route = "extract_answer"
        elif state.get("retry_count", 0) < config.MAX_LOCAL_RETRIES:
            target_route = "retry"
        else:
            print(f"[CHECK_LOCAL_RESULT] Retries exhausted, answer still insufficient.")
            return {
                "fallback_route": "extract_answer",
                "messages": [AIMessage(content="抱歉，知识库中没有找到与您问题相关的信息。")],
            }
        return {"fallback_route": target_route}

    except Exception as e:
        print(f"[CHECK_LOCAL_RESULT] LLM check failed: {e}, falling back to keyword matching")
        failure_keywords = [
            "NO_RELEVANT_CHUNKS",
            "NO_PARENT",
            "根据本地知识库，我无法找到相关信息",
            "本地知识库中没有",
            "无法在本地",
            "don't know",
            "不知道",
            "没有找到",
            "无法找到"
        ]
        if any(keyword in last_message for keyword in failure_keywords):
            if state.get("retry_count", 0) < config.MAX_LOCAL_RETRIES:
                return {"fallback_route": "retry"}
        return {"fallback_route": "extract_answer"}


def prepare_retry(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0) + 1
    hint = HumanMessage(
        content=(
            f"[Retry {retry_count}/{config.MAX_LOCAL_RETRIES}] "
            f"Previous search was insufficient. "
            f"Review your conversation history to see what you have already checked, "
            f"then try different search terms or different parent documents.\n"
            f"Question: {state['question']}"
        )
    )
    print(f"\n[RETRY] Attempt {retry_count}/{config.MAX_LOCAL_RETRIES} for: {state['question']}")
    return {
        "retry_count": retry_count,
        "messages": [hint],
    }
