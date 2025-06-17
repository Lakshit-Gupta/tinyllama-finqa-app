from fastapi import FastAPI, HTTPException,Request
from pydantic import BaseModel
import os
import boto3
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc
import sys
import json
import psutil
import shutil

# Update these paths for SageMaker
os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache/transformers'
os.environ['HF_HOME'] = '/tmp/cache/hf'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variables for model caching
_model = None
_tokenizer = None

class QuestionRequest(BaseModel):
    question: str

# Update MODEL_DIR
MODEL_DIR = "/tmp/model/adapter"
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "ap-south-1")
)

def download_model_from_s3():
    """Download LoRA adapter files from S3"""
    bucket_name = os.environ.get('MODEL_BUCKET', 'tinyllama-finqa-models')
    model_path = MODEL_DIR
    os.makedirs(model_path, exist_ok=True)
    
    files = [
        "adapter_config.json",
        "adapter_model.safetensors", 
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for file in files:
        try:
            s3_client.download_file(bucket_name, f"models/{file}", f"{model_path}/{file}")
            logger.info(f"Downloaded {file}")
        except Exception as e:
            logger.error(f"Error downloading {file}: {str(e)}")
    
    return model_path

# Clear cache before loading
cache_dirs = ["/tmp/cache", "/tmp/offload", "~/.cache"]
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        except:
            pass

def get_model():
    """Load model with maximum memory optimization"""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        logger.info("Using cached model")
        return _model, _tokenizer
    
    try:
        logger.info("Starting model loading...")
        
        # Download adapter files
        model_path = download_model_from_s3()
        
        # Load tokenizer
        base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        _tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            cache_dir="/tmp/cache",
            model_max_length=256
        )
        _tokenizer.pad_token = _tokenizer.eos_token
        logger.info("Tokenizer loaded")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="cpu",  # Force CPU only
            low_cpu_mem_usage=True,
            max_memory={"cpu": "12GB"},  # Limit memory usage
            offload_folder="/tmp/offload",
            cache_dir="/tmp/cache"
        )
        logger.info("Base model loaded")
        
        # Load PEFT adapter
        _model = PeftModel.from_pretrained(base_model, model_path)
        _model.eval()
        logger.info("PEFT adapter loaded")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return _model, _tokenizer
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def generate_answer(question):
    """Generate answer with memory optimization"""
    model, tokenizer = get_model()
    
    prompt = f"Q: {question}\nA:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=False
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.replace(prompt, "").strip()
    
    del inputs, outputs
    gc.collect()
    
    return answer

def check_resources():
    """Check system resources"""
    # Check disk space
    disk_usage = shutil.disk_usage('/')
    disk_free_gb = disk_usage.free / (1024**3)
    
    # Check memory
    memory = psutil.virtual_memory()
    memory_available_gb = memory.available / (1024**3)
    
    logger.info(f"Disk space free: {disk_free_gb:.2f} GB")
    logger.info(f"Memory available: {memory_available_gb:.2f} GB")
    logger.info(f"Memory used: {memory.percent}%")

# SageMaker health check endpoint
@app.get("/ping")
async def ping():
    """SageMaker health check endpoint"""
    return {"status": "healthy"}

# SageMaker inference endpoint
@app.post("/invocations")
async def invocations(request: dict = None):
    """SageMaker inference endpoint"""
    try:
        # Handle both direct dict and request body
        if request is None:
            raise HTTPException(status_code=400, detail="No request body provided")
        
        # Extract question from request
        if isinstance(request, dict):
            question = request.get("question", "")
        else:
            question = request
            
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
            
        logger.info(f"Processing question: {question}")
        answer = generate_answer(question)
        
        response = {
            "question": question,
            "answer": answer,
            "status": "success"
        }
        
        logger.info(f"Generated response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error in invocations: {str(e)}")
        return {
            "error": str(e),
            "status": "error"
        }

# Keep your original endpoints for testing
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = generate_answer(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))