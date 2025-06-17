# TinyLlama Financial Q&A API 🏦

A FastAPI-based backend service providing intelligent financial question-answering using a fine-tuned TinyLlama model. Deployed on HuggingFace Spaces with 16GB free RAM.

## 🚀 Live API Endpoint
https://lakshitgupta-tinyllama-finqa-api.hf.space

## 📋 API Documentation

### Base URL
https://lakshitgupta-tinyllama-finqa-api.hf.space

### Endpoints

#### 1. Ask Financial Question
**POST** `/ask`

Ask any financial question and get an intelligent response from our fine-tuned TinyLlama model.

**Request:**
```json
{
  "question": "What is revenue growth?"
}
Response:
{
  "question": "What is revenue growth?",
  "answer": "Revenue growth refers to the increase in a company's sales over a specific period, typically expressed as a percentage. It indicates how well a company is expanding its business and generating more income from its operations.",
  "status": "success"
}
Example Questions:

"What is dividend yield?"
"How do you calculate P/E ratio?"
"What is the difference between gross and net profit?"
"Explain compound annual growth rate (CAGR)"
2. Health Check
GET /health

Check if the API service is running and healthy.

Response:
{
  "status": "healthy",
  "model": "loaded"
}
3. System Status
GET /ping

Simple ping endpoint for monitoring.

Response:

{
  "status": "healthy"
}
⚡ Performance Notes
First Request: ~3-5 minutes (model loading from S3)
Subsequent Requests: ~2-5 seconds
Model Size: 2.2GB (TinyLlama base) + fine-tuned adapter
Memory Usage: ~3.5GB RAM
Concurrent Users: Optimized for mobile app usage
🧠 Model Information
Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Fine-tuning: LoRA (Low-Rank Adaptation) on financial Q&A dataset
Dataset: virattt/financial-qa-10K (10-K financial reports)
Training: 3 epochs with financial question-answer pairs
Specialization: Corporate finance, financial ratios, investment concepts
🔒 Rate Limits & Usage
Free Tier: Unlimited usage on HuggingFace Spaces
Concurrent Requests: Handles multiple users
Timeout: 30 seconds per request
Max Question Length: 512 tokens
🚨 Error Handling
Common Error Responses:
Invalid Request Format
{
  "status": "error",
  "error": "Invalid request format",
  "detail": "Question field is required"
}
Model Loading Error
{
  "status": "error",
  "error": "Model not available",
  "detail": "Please try again in a few minutes"
}
Server Error
{
  "status": "error",
  "error": "Internal server error",
  "detail": "Please contact support"
}
🛠 Technical Details
Framework: FastAPI
Model: TinyLlama + PEFT LoRA adapter
Deployment: HuggingFace Spaces (CPU Basic - 16GB RAM)
Storage: AWS S3 for model artifacts
Language: Python 3.9+
Dependencies: transformers, peft, torch, fastapi
📊 Status & Monitoring
Check service status:

Health: GET /health
System: GET /ping
Logs: Available in HuggingFace Spaces interface
🤝 Support
For technical issues or questions:

GitHub: [https://github.com/Lakshit-Gupta]
Email: [www.zxgupta12345678@gmail.com]
HuggingFace: @lakshitgupta
📜 License
MIT License - Feel free to use for personal and commercial projects.

🎯 Built for mobile developers who need reliable financial AI without infrastructure hassle.

🚀 Powered by TinyLlama, HuggingFace Spaces, and AWS S3.

