import asyncio
import tempfile
import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, UploadFile, File
from api.dependencies import get_document_manager

router = APIRouter()


@router.get("/documents")
async def list_documents(request: Request):
    dm = get_document_manager(request)
    files = dm.get_markdown_files()
    return {"files": files}


@router.post("/documents")
async def upload_documents(request: Request, files: List[UploadFile] = File(...)):
    dm = get_document_manager(request)

    tmp_dir = tempfile.mkdtemp()
    tmp_paths = []
    try:
        for f in files:
            suffix = Path(f.filename).suffix.lower()
            if suffix not in (".pdf", ".md", ".docx"):
                continue
            dest = os.path.join(tmp_dir, f.filename)
            content = await f.read()
            with open(dest, "wb") as fp:
                fp.write(content)
            tmp_paths.append(dest)

        if not tmp_paths:
            return {"added": 0, "skipped": 0}

        loop = asyncio.get_event_loop()
        added, skipped = await loop.run_in_executor(
            None, dm.add_documents, tmp_paths
        )
        return {"added": added, "skipped": skipped}
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


@router.delete("/documents")
async def clear_documents(request: Request):
    dm = get_document_manager(request)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, dm.clear_all)
    return {"status": "cleared"}
