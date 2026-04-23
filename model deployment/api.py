"""
api.py
======
FastAPI wrapper around the SageMaker endpoint.
Exposes a clean /ask REST API for the RAG pipeline to call.

Run locally:   uvicorn api:app --host 0.0.0.0 --port 8080
Deploy:        Docker → AWS ECR → ECS Fargate (free tier)
               OR just run on the same EC2 instance
"""

import os
import json
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "medqa-deepseek-v1")
AWS_REGION    = os.getenv("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = (
    "You are a medical assistant trained on clinical question answering. "
    "Answer concisely and accurately using medical knowledge. "
    "For multiple choice questions, state the answer and explain why."
)

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="MedQA AI Doc API",
    description="Fine-tuned DeepSeek medical QA model deployed on AWS SageMaker",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    context:  Optional[str] = None      # RAG pipeline injects context here
    max_tokens: Optional[int] = 256

class AnswerResponse(BaseModel):
    question: str
    answer:   str
    model:    str = "Hemachandiran/medqa-deepseek_v1"


# ─────────────────────────────────────────────
# Helper: Format prompt
# ─────────────────────────────────────────────
def build_prompt(question: str, context: Optional[str] = None) -> str:
    """Build Qwen2 chat-formatted prompt, optionally with RAG context."""
    if context:
        user_msg = (
            f"Use the following medical reference context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    else:
        user_msg = question

    # Qwen2 chat template
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": "medqa-deepseek_v1", "endpoint": ENDPOINT_NAME}


@app.get("/health")
def health():
    """Health check — verifies SageMaker endpoint is reachable."""
    try:
        sm = boto3.client("sagemaker", region_name=AWS_REGION)
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = desc["EndpointStatus"]
        return {"status": "ok", "endpoint_status": status}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Endpoint unreachable: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Main inference endpoint.
    
    - Without context: uses fine-tuned model knowledge only
    - With context: RAG pipeline pre-fills context from textbook retrieval
    
    Example:
        POST /ask
        {"question": "What is the treatment for Type 2 Diabetes?"}
    """
    prompt = build_prompt(request.question, request.context)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":     request.max_tokens,
            "temperature":        0.1,
            "repetition_penalty": 1.1,
            "do_sample":          True,
            "return_full_text":   False,
        }
    }

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())
        answer = result[0]["generated_text"] if isinstance(result, list) else str(result)
        answer = answer.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return AnswerResponse(question=request.question, answer=answer)