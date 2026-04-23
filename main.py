"""
Medical RAG Pipeline — FastAPI Endpoint
========================================
Exposes the MedicalRAGPipeline as a REST API.

Endpoints:
  POST /query        — Run the full RAG pipeline
  GET  /health       — Liveness check
  GET  /docs         — Auto-generated Swagger UI (FastAPI built-in)

Run locally:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Deploy on AWS (example):
  gunicorn api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the pipeline — assumes api.py lives in the same directory as rag_pipeline.py
from rag_pipeline import build_pipeline, MedicalRAGPipeline, SAGEMAKER_ENDPOINT, AWS_REGION


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="MedQA DeepSeek RAG API",
    description=(
        "Hybrid Adaptive + Corrective RAG pipeline over Harrison, Pathoma, "
        "and First Aid medical textbooks. Powered by a fine-tuned DeepSeek model "
        "on AWS SageMaker and Pinecone vector store."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pipeline singleton — loaded once on startup
# ─────────────────────────────────────────────

_pipeline: Optional[MedicalRAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    global _pipeline
    print("[API] Initialising pipeline...")
    _pipeline = build_pipeline(endpoint=SAGEMAKER_ENDPOINT, region=AWS_REGION)
    print("[API] Pipeline ready.")


def get_pipeline() -> MedicalRAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet. Try again shortly.")
    return _pipeline


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The medical question to answer.",
        example="What are the treatment options for Type 2 diabetes?",
    )


class SourceDocument(BaseModel):
    book: str = Field(description="Source textbook name.")
    relevance_score: float = Field(description="Keyword overlap relevance score (0–1).")
    section_type: Optional[str] = Field(default=None, description="Medical section type (e.g. treatment, diagnosis).")
    excerpt: str = Field(description="First 600 characters of the retrieved chunk.")


class QueryResponse(BaseModel):
    query: str = Field(description="The original question.")
    query_type: str = Field(description="Classified query type (factual / procedural / diagnostic / comparative / multi_hop).")
    answer: str = Field(description="Full answer: summary sentence + clinical detail from source excerpts.")
    num_docs: int = Field(description="Number of source documents used.")
    source_documents: List[SourceDocument] = Field(description="Retrieved and graded source documents.")
    raw_context: str = Field(description="Full formatted context passed to the model.")
    latency_ms: float = Field(description="Total pipeline latency in milliseconds.")


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    endpoint: str
    region: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Liveness check — confirms the API is running and pipeline is loaded."""
    return HealthResponse(
        status="ok",
        pipeline_ready=_pipeline is not None,
        endpoint=SAGEMAKER_ENDPOINT,
        region=AWS_REGION,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_pipeline(request: QueryRequest):
    """
    Run the full Hybrid Adaptive + Corrective RAG pipeline.

    **Pipeline steps:**
    1. Classify query type (factual / procedural / diagnostic / comparative / multi-hop)
    2. Adaptive routing — sets k, MMR flag, and query expansion flag
    3. Optional query expansion via SageMaker (complex queries only)
    4. Pinecone retrieval across 11,593 indexed medical textbook chunks
    5. Corrective RAG — keyword overlap grading, re-retrieval if quality is low
    6. Answer construction from retrieved context with source attribution
    """
    pipeline = get_pipeline()

    t0 = time.time()
    try:
        result = pipeline.run(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    latency_ms = round((time.time() - t0) * 1000, 1)

    # Serialise source documents
    source_documents = []
    for doc in result["source_docs"]:
        meta = doc.metadata
        book = (
            meta.get("book")
            or meta.get("source", "Unknown")
        )
        # Normalise book name (strip path + extension)
        import re, os
        book = os.path.basename(book).replace(".txt", "")

        source_documents.append(SourceDocument(
            book=book,
            relevance_score=float(meta.get("relevance_score", 0.0)),
            section_type=meta.get("section_type"),
            excerpt=doc.page_content[:600].strip(),
        ))

    return QueryResponse(
        query=result["query"],
        query_type=result["query_type"],
        answer=result["answer"],
        num_docs=result["num_docs"],
        source_documents=source_documents,
        raw_context=result["context"],
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────
# Global error handler
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected error: {str(exc)}"},
    )


# ─────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)