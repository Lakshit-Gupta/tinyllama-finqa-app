import os
import json
import logging
from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set ALL environment variables BEFORE importing any HF/torch libraries
os.environ["MODEL_BUCKET"] = "tinyllama-finqa-models" 
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/hf_home'
os.environ['XDG_CACHE_HOME'] = '/tmp/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_hub'
os.environ['TORCH_HOME'] = '/tmp/torch_home'
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Create necessary directories
for d in ['/tmp/transformers_cache', '/tmp/hf_home', '/tmp/cache', 
          '/tmp/huggingface_hub', '/tmp/torch_home', '/tmp/torch_extensions']:
    os.makedirs(d, exist_ok=True)

# Import app after environment setup
from app.main import app

# Lambda handler with error handling
handler = Mangum(app)

def lambda_handler(event, context):
    """Process API requests with error handling"""
    try:
        return handler(event, context)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e), 'status': 'error'})
        }