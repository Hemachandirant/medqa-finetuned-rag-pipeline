import boto3
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "medqa-deepseek-v2")

runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps({
        "inputs": "What is diabetes?",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    })
)

result = response["Body"].read().decode()
print(result)