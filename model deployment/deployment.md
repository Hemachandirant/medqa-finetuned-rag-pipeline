# AWS Deployment Guide — MedQA DeepSeek v1

Model: `Hemachandiran/medqa-deepseek_v1` (Qwen2, 3.55 GB)  
Architecture: SageMaker (model inference) + FastAPI (API wrapper)

---

## Architecture

```
Client (curl / RAG pipeline)
        │
        ▼
  FastAPI /ask  (runs on render)
        │
        ▼
  AWS SageMaker Endpoint  (ml.g4dn.xlarge — T4 GPU)
        │
        ▼
  medqa-deepseek_v1 (pulled from HuggingFace Hub)
```

---

## Step 1 — Prerequisites

```bash
pip install boto3 sagemaker awscli

# Configure AWS credentials
aws configure
# Prompts for:
#   AWS Access Key ID:     (from AWS Console → IAM → Your User → Security Credentials)
#   AWS Secret Access Key: (same place)
#   Default region:        us-east-1
#   Default output:        json
```

---

## Step 2 — Create IAM Role (AWS Console, one-time)

1. Go to: https://console.aws.amazon.com/iam/home#/roles
2. Click **Create role**
3. Trusted entity: **AWS Service → SageMaker**
4. Attach permission: **AmazonSageMakerFullAccess**
5. Role name: `SageMakerExecutionRole`
6. Create role → copy the ARN:
   `arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole`

---

## Step 3 — Set Environment Variables

```bash
export HF_TOKEN="hf_your_token_here"          # huggingface.co/settings/tokens
export SAGEMAKER_ROLE="arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
export AWS_REGION="us-east-1"
```

Edit `deploy_to_sagemaker.py` line 27-28 OR set via env vars above.

---

## Step 4 — Deploy Model to SageMaker

```bash
python deploy_to_sagemaker.py --deploy
```

**What happens:**
- SageMaker pulls `Hemachandiran/medqa-deepseek_v1` from HF Hub (~3.55 GB)
- Spins up a `ml.g4dn.xlarge` instance (T4 GPU, 16 GB VRAM)
- Starts TGI (Text Generation Inference) server
- Creates endpoint: `medqa-deepseek-v1`

**Time:** ~8–12 minutes  
**Cost:** ~$0.736/hr — DELETE when not in use!

---

## Step 5 — Test the SageMaker Endpoint

```bash
python deploy_to_sagemaker.py --test

# Custom question:
python deploy_to_sagemaker.py --test --question "What are the symptoms of pulmonary embolism?"
```

Expected output:
```
[Test] Question: What are the symptoms of pulmonary embolism?
[Test] Answer:
Pulmonary embolism presents with sudden onset dyspnea, pleuritic chest pain,
tachycardia, and hemoptysis. Hypoxia is common. Massive PE can cause
haemodynamic instability and syncope...
```

---

## Step 6 — Run the FastAPI Wrapper

```bash
# Install dependencies
pip install -r requirements.txt

# Run API (connects to your SageMaker endpoint)
SAGEMAKER_ENDPOINT=medqa-deepseek-v1 uvicorn api:app --host 0.0.0.0 --port 8080
```

Test it:
```bash
# Simple question
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the first-line treatment for hypertension?"}'

# With RAG context (used by rag_pipeline.py)
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the dosing of metformin in CKD?",
    "context": "Metformin is contraindicated when eGFR < 30 mL/min/1.73m². Dose reduction recommended for eGFR 30-45..."
  }'
```

API docs: http://localhost:8080/docs

---

## Step 7 — (Optional) Deploy FastAPI to AWS EC2 Free Tier

```bash
# Launch t2.micro EC2 (free tier — no GPU needed, just the wrapper)
# 1. AWS Console → EC2 → Launch Instance
# 2. AMI: Ubuntu 22.04
# 3. Instance: t2.micro (free tier)
# 4. Security group: open port 8080

# SSH into instance
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# On EC2:
sudo apt update && sudo apt install -y python3-pip
pip install fastapi uvicorn boto3 sagemaker pydantic

# Copy api.py to EC2, then:
SAGEMAKER_ENDPOINT=medqa-deepseek-v1 \
AWS_REGION=us-east-1 \
nohup uvicorn api:app --host 0.0.0.0 --port 8080 &

# Your public API:
# http://YOUR_EC2_IP:8080/ask
```

---

## Step 8 — Connect RAG Pipeline to This API

In `rag_pipeline.py`, update the LLM section to call your API instead of loading locally:

```python
import requests

class MedQAAPIClient:
    def __init__(self, api_url: str):
        self.api_url = api_url  # http://YOUR_EC2_IP:8080

    def invoke(self, prompt_dict: dict) -> str:
        response = requests.post(
            f"{self.api_url}/ask",
            json={
                "question": prompt_dict["question"],
                "context":  prompt_dict.get("context", ""),
            }
        )
        return response.json()["answer"]

# Replace load_llm() with:
llm = MedQAAPIClient("http://YOUR_EC2_IP:8080")
```

---

## Cost Summary

| Component | Instance | Cost |
|-----------|----------|------|
| SageMaker Endpoint | ml.g4dn.xlarge | ~$0.736/hr |
| EC2 API Wrapper | t2.micro | FREE (750 hrs/mo free tier) |
| Data Transfer | — | Negligible |

**⚠️ Important:** Delete SageMaker endpoint when not in use:
```bash
python deploy_to_sagemaker.py --delete
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ResourceLimitExceeded` | Request GPU quota increase: AWS Console → Service Quotas → SageMaker → `ml.g4dn.xlarge for endpoint usage` → Request increase to 1 |
| `ModelError: 503` | Endpoint still loading — wait 2 min and retry |
| `AccessDeniedException` | IAM role missing — redo Step 2 |
| `HF_TOKEN invalid` | Regenerate at huggingface.co/settings/tokens |