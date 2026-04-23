"""
debug_endpoint.py
Sends a direct request to SageMaker and prints the RAW response
so we can see exactly what DeepSeek is returning before any processing.
"""

import os
import json
import boto3
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ENDPOINT  = os.getenv("SAGEMAKER_ENDPOINT_NAME", "medqa-deepseek-v2")
REGION    = os.getenv("AWS_REGION", "ap-south-1")

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# ── Test 1: Plain question, no system prompt ──────────────────────────
print("\n" + "="*60)
print("TEST 1: Plain prompt (no system, no context)")
print("="*60)

payload1 = {
    "inputs": "What is the treatment for Type 2 diabetes?\n",
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "return_full_text": False,
    }
}

resp1 = runtime.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="application/json",
    Body=json.dumps(payload1),
)
raw1 = json.loads(resp1["Body"].read().decode())
print("RAW RESPONSE:")
print(json.dumps(raw1, indent=2))


# ── Test 2: Chat format ───────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: Chat format with im_start tokens")
print("="*60)

formatted2 = (
    "<|im_start|>system\n"
    "You are a helpful medical assistant.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "What is the first-line treatment for Type 2 diabetes? Explain in detail.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

payload2 = {
    "inputs": formatted2,
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "return_full_text": False,
    }
}

resp2 = runtime.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="application/json",
    Body=json.dumps(payload2),
)
raw2 = json.loads(resp2["Body"].read().decode())
print("RAW RESPONSE:")
print(json.dumps(raw2, indent=2))


# ── Test 3: MedQA fine-tune format (how it was trained) ──────────────
print("\n" + "="*60)
print("TEST 3: MedQA fine-tune format (question + options)")
print("="*60)

formatted3 = (
    "<|im_start|>user\n"
    "Question: A 55-year-old patient is diagnosed with Type 2 diabetes. "
    "What is the first-line pharmacological treatment?\n"
    "A. Insulin\n"
    "B. Metformin\n"
    "C. Sulfonylurea\n"
    "D. GLP-1 agonist\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

payload3 = {
    "inputs": formatted3,
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "return_full_text": False,
    }
}

resp3 = runtime.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="application/json",
    Body=json.dumps(payload3),
)
raw3 = json.loads(resp3["Body"].read().decode())
print("RAW RESPONSE:")
print(json.dumps(raw3, indent=2))

print("\n" + "="*60)
print("DONE — share all 3 outputs above")
print("="*60)