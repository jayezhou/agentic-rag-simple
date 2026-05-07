import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api.dependencies import get_chat_interface

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@router.post("/chat")
async def chat_endpoint(body: ChatRequest, request: Request):
    ci = get_chat_interface(request)

    async def event_generator():
        async for event in ci.async_stream_chat(body.message, body.thread_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
