
import asyncio
import json
import os
import re
import time
from io import BytesIO
from uuid import uuid4
import numpy as np
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="DocuStream", version="1.0.0")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Groq offers an API compatible with the OpenAI Python client.
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv(
    "GROQ_MODEL", "llama-3.3-70b-versatile"
)

client = (
    AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    if api_key
    else None
)

# In-memory storage: documents disappear when the app restarts.
documents = {}


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Summary(BaseModel):
    title: str
    document_type: str
    key_entities: list[str]
    one_line_summary: str


@app.get("/")
async def home():
    return {"message": "Welcome to DocuStream!", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


def get_client():
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured.",
        )
    return client


def split_into_chunks(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + size])
        if chunk.strip():
            chunks.append(chunk)

        if start + size >= len(words):
            break

        start += size - overlap

    return chunks


def retrieve_chunks(question, chunks, chunk_embeddings, top_k=3):
    if not chunks:
        return []

    question_vector = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    scores = np.dot(chunk_embeddings, question_vector)

    best_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(index), chunks[index])
        for index in best_indices
    ]

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        reader = PdfReader(BytesIO(content))

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported.",
            )

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No extractable text found. "
                    "Scanned PDFs are not supported."
                ),
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF file.",
        ) from exc
    finally:
        await file.close()

    chunks = split_into_chunks(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="The PDF contains no usable text.",
        )
    chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True,
)

    # Each upload gets a new ID; duplicate uploads are stored separately.
    document_id = str(uuid4())
    documents[document_id] = {
    "filename": file.filename or "document.pdf",
    "chunks": chunks,
    "embeddings": chunk_embeddings,
}
    return {
        "document_id": document_id,
        "num_chunks": len(chunks),
        "status": "ready",
    }


@app.post("/documents/{document_id}/query")
async def query_document(document_id: str, body: Question):
    if document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    ai = get_client()
    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank.",
        )

    chunks = documents[document_id]["chunks"]
    chunk_embeddings = documents[document_id]["embeddings"]
    selected = retrieve_chunks(question, chunks, chunk_embeddings)
    context = "\n\n".join(
        f"[Chunk {index}]\n{chunk}"
        for index, chunk in selected
    )

    async def event_stream():
        started = time.perf_counter()

        try:
            response = await ai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using the provided document context. "
                            "If the answer is not in the context, say so. "
                            "Do not invent facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Document context:\n{context}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                stream=True,
                temperature=0.2,
            )

            async for part in response:
                if not part.choices:
                    continue

                token = part.choices[0].delta.content
                if token:
                    # Each token is sent as it arrives.
                    data = json.dumps({"token": token})
                    yield f"data: {data}\n\n"

            metadata = {
                "chunks_used": [index for index, _ in selected],
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000
                ),
            }
            yield (
                "event: sources\n"
                f"data: {json.dumps(metadata)}\n\n"
            )

        except asyncio.CancelledError:
            # The client disconnected; stop generating.
            raise
       
        except Exception as exc:
            print(
                "AI request failed:",
                type(exc).__name__,
                str(exc),
            )
            yield (
                "event: error\n"
                + 'data: {"detail":"Answer generation failed."}\n\n'
            )


    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/documents/{document_id}/summary")
async def summarize_document(document_id: str):
    if document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    ai = get_client()
    chunks = documents[document_id]["chunks"]

    # Limit summary input to avoid sending a huge document at once.
    context = "\n\n".join(chunks[:10])

    prompt = (
        "Summarize the following document. Return a JSON object with "
        "exactly these fields: title (string), document_type "
        "(one of invoice, report, resume, other), key_entities "
        "(array of strings), and one_line_summary (string). "
        "Use only information supported by the document. "
        "Do not invent missing information.\n\n"
        f"Document:\n{context}"
    )

    for attempt in range(2):
        try:
            response = await ai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return valid JSON only. "
                            "Do not include Markdown fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            raw = response.choices[0].message.content or ""
            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                raw.strip(),
            )

            parsed = json.loads(raw)
            summary = Summary.model_validate(parsed)

            allowed_types = {"invoice", "report", "resume", "other"}
            if summary.document_type not in allowed_types:
                raise ValueError("Invalid document_type")

            return summary.model_dump()

        except (json.JSONDecodeError, ValidationError, ValueError):
            if attempt == 0:
                continue
            raise HTTPException(
                status_code=502,
                detail="The AI returned invalid summary JSON.",
            )
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="Summary generation failed.",
            )

    raise HTTPException(
        status_code=502,
        detail="Could not generate a valid summary.",
    )
