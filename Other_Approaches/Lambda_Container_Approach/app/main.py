from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import boto3
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc
import sys

# Set environment variables first
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/hf_home'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variables for model caching
_model = None
_tokenizer = None

class QuestionRequest(BaseModel):
    question: str

MODEL_DIR = "/tmp/model"
s3_client = boto3.client("s3")

def download_model_from_s3():
    """Download LoRA adapter files from S3"""
    bucket_name = os.environ.get('MODEL_BUCKET')
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
        except Exception as e:
            logger.error(f"Error downloading {file}: {str(e)}")
    
    return model_path

def get_model():
    """Load model with maximum memory optimization"""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        # Download adapter files
        model_path = download_model_from_s3()
        
        # Load tokenizer only
        base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        _tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            cache_dir="/tmp/cache",
            model_max_length=256  # Reduced
        )
        _tokenizer.pad_token = _tokenizer.eos_token
        
        # Load base model with aggressive optimization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,  # Half precision
            device_map="cpu",
            low_cpu_mem_usage=True,
            cache_dir="/tmp/cache",
            use_cache=False  # Disable KV cache
        )
        
        # Load PEFT adapter
        _model = PeftModel.from_pretrained(base_model, model_path)
        _model.eval()
        
        # Force cleanup
        gc.collect()
        
        return _model, _tokenizer
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def generate_answer(question):
    """Generate answer with memory optimization"""
    model, tokenizer = get_model()
    
    # Shorter prompt
    prompt = f"Q: {question}\nA:"
    
    # Tokenize with limits
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,  # Reduced
        truncation=True,
        padding=False  # No padding
    )
    
    # Generate with strict limits
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Very limited
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False  # No caching
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.replace(prompt, "").strip()
    
    # Cleanup
    del inputs, outputs
    gc.collect()
    
    return answer

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