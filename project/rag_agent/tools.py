import os
import requests
from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager


class ToolFactory:

    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks from the local knowledge base.

        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            # Step 1: Broad search for candidates
            candidates = self.collection.similarity_search(query, k=limit * 3)
            if not candidates:
                return "NO_RELEVANT_CHUNKS"

            # Step 2: Rerank using DashScope API
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                results = candidates[:limit]
            else:
                url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gte-rerank",
                    "input": {
                        "query": query,
                        "documents": [doc.page_content for doc in candidates]
                    },
                    "parameters": {"top_n": limit}
                }

                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

                top_results = []
                for item in result.get("output", {}).get("results", []):
                    idx = item["index"]
                    top_results.append(candidates[idx])
                results = top_results

            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunk by its ID. Use this when a child chunk is relevant
        but its content is truncated — the parent chunk contains the full context.

        Args:
            parent_id: Parent chunk ID to retrieve (from search_child_chunks results)
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    def get_local_tools(self) -> List:
        """Create and return local RAG tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        return [search_tool, retrieve_tool]
