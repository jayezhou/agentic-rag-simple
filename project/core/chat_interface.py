from typing import AsyncIterator
from langchain_core.messages import HumanMessage


class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system

    async def async_stream_chat(
        self, message: str, thread_id: str | None = None
    ) -> AsyncIterator[dict]:
        """Yield SSE event dicts. Caller serializes to `data: <json>\\n\\n`."""
        if not self.rag_system.agent_graph:
            yield {"type": "error", "detail": "System not initialized"}
            return

        if thread_id is None:
            self.rag_system.reset_thread()
            thread_id = self.rag_system.thread_id

        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for event in self.rag_system.agent_graph.astream_events(
                {"messages": [HumanMessage(content=message.strip())]},
                config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node == "aggregate":
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            yield {"type": "token", "content": chunk}
                elif kind == "on_chain_end" and event["name"] in (
                    "summarize", "analyze_rewrite", "process_question", "aggregate"
                ):
                    yield {"type": "node_end", "node": event["name"]}

            state = self.rag_system.agent_graph.get_state(config)
            if "human_input" in (state.next or []):
                msgs = state.values.get("messages", [])
                clarification = msgs[-1].content if msgs else "Please clarify your question."
                yield {"type": "interrupt", "thread_id": thread_id, "message": clarification}
            else:
                msgs = state.values.get("messages", [])
                final = msgs[-1].content if msgs else ""
                yield {"type": "done", "thread_id": thread_id, "content": final}

        except Exception as e:
            yield {"type": "error", "detail": str(e)}

    async def chat(self, message, history):

        if not self.rag_system.agent_graph:
            return "System not initialized!"

        try:
            result = await self.rag_system.agent_graph.ainvoke(
                {"messages": [HumanMessage(content=message.strip())]},
                self.rag_system.get_config()
            )
            return result["messages"][-1].content

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_session(self):
        self.rag_system.reset_thread()
