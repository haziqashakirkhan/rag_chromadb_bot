from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ingest_service import IngestionService
from app.rag_service import RAGService


app = FastAPI(
    title="Groq + ChromaDB RAG API",
    version="1.0.0",
)
BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt", ".md"}

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

rag_service: Optional[RAGService] = None


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
    )

    source: Optional[str] = None
    model: Optional[str] = None


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.on_event("startup")
def startup() -> None:
    global rag_service
    print("Starting RAG service...")

    try:
        rag_service = RAGService()
        print("RAG service started successfully.")

    except Exception as exc:
        print(f"RAG startup failed: {exc}")
        raise

@app.get("/health") 
def health():
    if rag_service is None:
        return {
            "status": "starting"
        }

    return {
        "status": "ok",
        "indexed_chunks": (
            rag_service.store.count()
        ),
    }


@app.post("/api/upload")
def upload_document(
    file: UploadFile = File(...),
):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF, TXT, or MD file.",
        )

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Files must be 15 MB or smaller.",
        )

    try:
        with TemporaryDirectory(prefix="rag-upload-") as directory:
            upload_path = Path(directory) / filename
            upload_path.write_bytes(contents)
            result = IngestionService().ingest(
                directory=directory,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not index {filename}: {exc}",
        ) from exc

    return {
        "filename": filename,
        **result,
    }


@app.post("/rag/query")
def query_rag(
    payload: QuestionRequest
):
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    try:
        return rag_service.ask(
            question=payload.question,
            top_k=payload.top_k,
            source=payload.source,
            model=payload.model,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
