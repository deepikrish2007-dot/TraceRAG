from pathlib import Path
import traceback

import chromadb
import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.src.pdf_loader import load_pdf
from backend.src.chunker import chunk_pages
from backend.src.embedding_store import add_chunks, DB_DIR, COLLECTION_NAME
from backend.src.answer import answer_question


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

chroma_client = chromadb.PersistentClient(path=str(DB_DIR))


def get_collection():
    """Retrieve the current ChromaDB collection by name."""
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------
# 2. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="TraceRAG API",
    description="Research paper question-answering API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. REQUEST MODELS
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# --------------------------------------------------
# 4. HOME / HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    try:
        collection = get_collection()
        return {
            "message": "TraceRAG API is running",
            "status": "healthy",
            "indexed_chunks": collection.count(),
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Database is unavailable: {exc}",
        ) from exc


# --------------------------------------------------
# 5. LIST DOCUMENTS
# --------------------------------------------------

@app.get("/documents")
def list_documents():
    try:
        documents = []

        for path in DATA_DIR.iterdir():
            if path.is_file() and path.suffix.lower() == ".pdf":
                documents.append({
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "size_kb": round(path.stat().st_size / 1024, 2),
                })

        return {
            "documents": documents,
            "total": len(documents),
            "indexed_chunks": get_collection().count(),
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not list documents: {exc}",
        ) from exc


# --------------------------------------------------
# 6. VIEW A PDF (used by source citations)
# --------------------------------------------------

@app.get("/documents/{filename}/view")
def view_document(filename: str, page: int = 1):
    """Serve an uploaded PDF inline; the browser can open a page with #page=N."""
    safe_filename = Path(filename).name

    if safe_filename != filename or safe_filename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if page < 1:
        raise HTTPException(
            status_code=400,
            detail="Page must be 1 or greater.",
        )

    file_path = DATA_DIR / safe_filename

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found.")

    # Validate that the file is a readable PDF, but do not reject the whole
    # PDF request just because citation metadata contains an out-of-range page.
    # The browser's #page=N fragment handles navigation after the PDF is served.
    # This avoids returning an API 404 for an otherwise valid PDF.
    try:
        with fitz.open(str(file_path)) as pdf:
            page_count = len(pdf)
            if page_count < 1:
                raise HTTPException(
                    status_code=422,
                    detail="This PDF contains no pages.",
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not read this PDF.",
        ) from exc

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=safe_filename,
        content_disposition_type="inline",
    )


# --------------------------------------------------
# 7. UPLOAD AND INDEX PDF
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    filename = Path(file.filename).name

    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    destination = DATA_DIR / filename
    temporary_path = DATA_DIR / f"{filename}.uploading"

    try:
        # Save the uploaded file temporarily.
        total_size = 0

        with temporary_path.open("wb") as output:
            while True:
                block = await file.read(1024 * 1024)

                if not block:
                    break

                total_size += len(block)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 25 MB upload limit.",
                    )

                output.write(block)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Validate PDF content, not just the filename extension.
        try:
            with fitz.open(str(temporary_path)) as pdf:
                if pdf.is_encrypted:
                    raise HTTPException(
                        status_code=422,
                        detail="Password-protected PDFs are not supported.",
                    )

                if len(pdf) == 0:
                    raise HTTPException(
                        status_code=422,
                        detail="The PDF contains no pages.",
                    )

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The uploaded file is not a valid, readable PDF. "
                    "Please select the original PDF file."
                ),
            ) from exc

        # Move to the final PDF path before calling the PDF loader.
        temporary_path.replace(destination)

        # Extract PDF text.
        pages = load_pdf(str(destination))

        if not pages:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text was found in the PDF. "
                    "It may be scanned and require OCR."
                ),
            )

        # Split extracted pages into chunks.
        chunks = chunk_pages(pages)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text chunks could be created from this PDF.",
            )

        # Index chunks using the shared embedding store.
        indexed_count = add_chunks(chunks)

        if not isinstance(indexed_count, int) or indexed_count <= 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The PDF was read, but no chunks were indexed. "
                    "Check embedding_store.py and the backend terminal."
                ),
            )

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": filename,
            "pages_extracted": len(pages),
            "chunks_created": len(chunks),
            "chunks_indexed": indexed_count,
            "total_indexed_chunks": get_collection().count(),
        }

    except HTTPException:
        raise

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {exc}",
        ) from exc

    finally:
        await file.close()

        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


# --------------------------------------------------
# 8. ASK A QUESTION
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        if get_collection().count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No indexed passages found. Upload a PDF first.",
            )

        answer = answer_question(question)
        return {"answer": answer}

    except HTTPException:
        raise

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate an answer: {exc}",
        ) from exc


# --------------------------------------------------
# 9. DELETE A DOCUMENT
# --------------------------------------------------

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    safe_filename = Path(filename).name

    if safe_filename != filename or safe_filename in {"", ".", ".."}:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    file_path = DATA_DIR / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        collection = get_collection()

        # Remove chunks belonging to this PDF.
        collection.delete(where={"source": safe_filename})
        file_path.unlink()

        return {
            "message": "Document and its indexed chunks deleted.",
            "filename": safe_filename,
            "remaining_chunks": get_collection().count(),
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not completely delete document: {exc}",
        ) from exc
