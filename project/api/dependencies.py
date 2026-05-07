import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from core.rag_system import RAGSystem
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_system = RAGSystem()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, rag_system.initialize)
    app.state.rag_system = rag_system
    app.state.chat_interface = ChatInterface(rag_system)
    app.state.document_manager = DocumentManager(rag_system)
    yield


def get_chat_interface(request: Request) -> ChatInterface:
    return request.app.state.chat_interface


def get_document_manager(request: Request) -> DocumentManager:
    return request.app.state.document_manager
