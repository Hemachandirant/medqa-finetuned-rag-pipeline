"""
deploy_to_sagemaker.py — Fixed for Qwen2 / medqa-deepseek_v1 (3.57 GB)
"""

import boto3
import sagemaker
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# Load environment variables from .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# ─────────────────────────────────────────────
# CONFIG (load from .env)
# ─────────────────────────────────────────────
HF_MODEL_ID   = os.getenv("HF_MODEL_ID", "Hemachandiran/medqa-deepseek_v1")
HF_TOKEN      = os.getenv("HF_TOKEN", "")
IAM_ROLE_ARN  = os.getenv("IAM_ROLE_ARN", "")
AWS_REGION    = os.getenv("AWS_REGION", "ap-south-1")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "medqa-deepseek-v2")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.g4dn.xlarge")
# ─────────────────────────────────────────────


def get_sagemaker_session():
    boto_session      = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    return sagemaker_session


def cleanup_existing():
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    for resource, fn in [
        ("endpoint",        lambda: sm.delete_endpoint(EndpointName=ENDPOINT_NAME)),
        ("endpoint-config", lambda: sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)),
        ("model",           lambda: sm.delete_model(ModelName=ENDPOINT_NAME)),
    ]:
        try:
            fn()
            print(f"   🗑  Deleted old {resource}")
        except Exception:
            pass


def deploy_model():
    print(f"[Deploy] Cleaning up any existing resources...")
    cleanup_existing()

    session = get_sagemaker_session()

    # Auto-fetch correct TGI image URI for ap-south-1
    image_uri = get_huggingface_llm_image_uri(
        backend="huggingface",
        region=AWS_REGION,
    )
    print(f"[Deploy] Image URI : {image_uri}")
    print(f"[Deploy] Model     : {HF_MODEL_ID}")
    print(f"[Deploy] Region    : {AWS_REGION}")
    print(f"[Deploy] Instance  : {INSTANCE_TYPE}")
    print(f"[Deploy] Endpoint  : {ENDPOINT_NAME}")
    print(f"[Deploy] This takes ~5-8 minutes...\n")

    huggingface_model = HuggingFaceModel(
        model_data=None,
        env={
            "HF_MODEL_ID":              HF_MODEL_ID,
            "HF_TOKEN":                 HF_TOKEN,
            "HUGGING_FACE_HUB_TOKEN":   HF_TOKEN,
            "HF_TASK":                  "text-generation",
            "SM_NUM_GPUS":              "1",
            "MAX_INPUT_LENGTH":         "2048",
            "MAX_TOTAL_TOKENS":         "4096",
            "MAX_BATCH_PREFILL_TOKENS": "4096",
            "TRUST_REMOTE_CODE":        "true",
            "USE_FLASH_ATTENTION":      "false",
        },
        role=IAM_ROLE_ARN,
        image_uri=image_uri,
        sagemaker_session=session,
    )

    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        container_startup_health_check_timeout=900,
    )

    print(f"\n✅ Deployed successfully!")
    print(f"   Endpoint : {ENDPOINT_NAME}")
    print(f"   Region   : {AWS_REGION}")
    print(f"   URL      : https://runtime.sagemaker.{AWS_REGION}.amazonaws.com/endpoints/{ENDPOINT_NAME}/invocations")
    print(f"\n   ⚠️  Run --delete when done to avoid charges!\n")
    return predictor


def test_endpoint(question: str = None):
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

    test_q = question or (
        "A 45-year-old patient presents with polyuria, polydipsia, "
        "and fasting glucose of 140 mg/dL. What is the most likely diagnosis?"
    )

    prompt = (
        "<|im_start|>system\n"
        "You are a medical assistant. Answer concisely and accurately.<|im_end|>\n"
        f"<|im_start|>user\n{test_q}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":     512,
            "temperature":        0.1,
            "repetition_penalty": 1.1,
            "do_sample":          True,
            "return_full_text":   False,
        }
    }

    print(f"\n[Test] Question: {test_q}")
    print("[Test] Calling endpoint...\n")

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode())
    answer = result[0]["generated_text"] if isinstance(result, list) else str(result)
    print(f"Answer:\n{answer.strip()}\n")
    return answer


def check_status():
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        desc   = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = desc["EndpointStatus"]
        print(f"Endpoint status: {status}")
        if status == "Failed":
            print(f"Reason: {desc.get('FailureReason', 'unknown')}")
    except Exception as e:
        print(f"Endpoint not found: {e}")


def delete_endpoint():
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    for resource, fn in [
        ("endpoint",        lambda: sm.delete_endpoint(EndpointName=ENDPOINT_NAME)),
        ("endpoint-config", lambda: sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)),
        ("model",           lambda: sm.delete_model(ModelName=ENDPOINT_NAME)),
    ]:
        try:
            fn()
            print(f"✅ Deleted {resource}")
        except Exception as e:
            print(f"   {resource}: {e}")
    print("\n✅ All deleted — no more charges!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--question", type=str)
    args = parser.parse_args()

    if   args.deploy:  deploy_model()
    elif args.test:    test_endpoint(args.question)
    elif args.status:  check_status()
    elif args.delete:  delete_endpoint()
    else: print("Usage: --deploy | --test | --status | --delete")