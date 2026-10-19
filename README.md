# DocuStream — Document Q&A Microservice

DocuStream is a document question-answering microservice that allows users to upload PDF files, ask questions about their contents, and generate structured summaries. It uses semantic search to retrieve relevant text chunks and a large language model (LLM) to generate answers.

## Features

* **PDF Upload:** Accepts PDF files, extracts their text, splits the text into overlapping chunks, and generates embeddings.
* **Semantic Retrieval:** Uses sentence-transformer embeddings and cosine similarity to identify relevant chunks for a user's question.
* **Streaming Question Answering:** Streams generated answer tokens using Server-Sent Events (SSE).
* **Source Metadata:** Sends a final SSE event containing the retrieved chunk indices and response latency.
* **Structured Summaries:** Generates a JSON summary containing the document title, document type, key entities, and a one-line summary.
* **Error Handling:** Handles missing document IDs, invalid requests, and PDFs that do not contain extractable text.
* **Docker Support:** Packages the application and its dependencies into a Docker image and runs it using Docker Compose.

## Technology Stack

* **Python 3.11**
* **FastAPI** — REST API framework
* **PyPDF** — PDF text extraction
* **Sentence Transformers** — Text embeddings using `all-MiniLM-L6-v2`
* **NumPy** — Similarity calculations
* **Groq API** — LLM-powered question answering and summaries
* **Pydantic** — Request validation and summary schema validation
* **Docker and Docker Compose** — Containerization and deployment

## Architecture

1. **Upload:** The client uploads a PDF through `POST /documents`.
2. **Extraction:** PyPDF extracts the text from the PDF.
3. **Chunking:** The text is divided into chunks of approximately 500 words, with a 50-word overlap.
4. **Embedding:** Sentence Transformers generates a vector embedding for each chunk.
5. **Storage:** The document's chunks and embeddings are stored in memory and associated with a generated document ID.
6. **Retrieval:** When a question is submitted, its embedding is compared with the stored chunk embeddings using cosine similarity.
7. **Generation:** The most relevant chunks are included in the prompt sent to the Groq-hosted language model.
8. **Streaming:** The answer is returned incrementally using SSE, followed by a `sources` event containing retrieval metadata.
9. **Summarization:** The summary endpoint requests structured JSON from the language model and validates the response before returning it.

## API Endpoints

### 1. Upload a PDF

`POST /documents`

**Request:** Multipart form data containing a PDF file in the `file` field.

**Example response:**

```json
{
  "document_id": "generated-document-id",
  "num_chunks": 35,
  "status": "ready"
}
```

### 2. Ask a Question

`POST /documents/{document_id}/query`

**Request body:**

```json
{
  "question": "What are the main points in this document?"
}
```

**Response:** A stream of SSE events containing answer tokens, followed by a final `sources` event with the retrieved chunk indices and latency in milliseconds.

### 3. Summarize a Document

`GET /documents/{document_id}/summary`

**Example response structure:**

```json
{
  "title": "Example Document",
  "document_type": "report",
  "key_entities": ["Example Entity"],
  "one_line_summary": "A concise summary of the document."
}
```

The `document_type` field supports `invoice`, `report`, `resume`, and `other`.

## Prerequisites

* Python 3.11 or later for running locally
* Docker Desktop with Docker Compose for containerized execution
* A Groq API key

## Configuration

Create a `.env` file in the project root using `.env.example` as a template.

Set the following environment variables:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Never commit the real `.env` file or expose your API key publicly.

## Run with Docker

1. Install and start Docker Desktop.
2. Open a terminal in the project directory.
3. Build and start the service:

```bash
docker compose up --build
```

4. Open the interactive API documentation at:

`http://localhost:8000/docs`

To stop the service, press `Ctrl + C`. To run it in the background, use `docker compose up -d`; stop it with `docker compose down`.

The embedding model may need to download the first time the container starts, so an internet connection is required during the initial setup.

## Run Locally

Create and activate a virtual environment, then install the dependencies.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure the `.env` file, then start the application:

```powershell
python -m uvicorn main:app --reload
```

Open `http://localhost:8000/docs` to explore the endpoints.

## Design Decisions

### 1. In-memory document storage

Chunks and embeddings are stored in memory to keep the implementation simple and avoid requiring an external database. This suits a small demonstration but means documents are lost when the application restarts.

### 2. Semantic retrieval using sentence embeddings

The `all-MiniLM-L6-v2` model converts text chunks and questions into vector embeddings. Normalized embeddings allow cosine similarity to be calculated efficiently using NumPy. This retrieves semantically relevant text rather than relying only on exact keyword matches.

### 3. Server-Sent Events for streaming

SSE allows the API to deliver generated answer tokens incrementally over an HTTP response. A final event communicates source chunk indices and latency without waiting for the entire answer to be displayed.

## Limitations

* Documents and embeddings are held in memory and are not persistent.
* Retrieval is limited to the top three chunks.
* Scanned PDFs without extractable text require OCR, which is not implemented.
* The embedding model requires an initial download unless it is already cached.
* The service does not currently include authentication, user-specific document access, or persistent vector storage.

## Future Improvements

* Add persistent storage with Chroma, FAISS, or a database.
* Add OCR support for scanned PDFs.
* Introduce automated tests for uploads, retrieval, streaming, summaries, and error cases.
* Add authentication, request limits, and stronger file validation.
* Improve retrieval using metadata filtering, reranking, and configurable chunking.
* Add monitoring and structured logging for production deployment.

## Security

* API credentials are loaded from environment variables.
* `.env` is excluded from the Docker build context through `.dockerignore` and should also be ignored by Git.
* `.env.example` provides a placeholder configuration for other developers.

## License

Add an appropriate license before distributing the project publicly.
