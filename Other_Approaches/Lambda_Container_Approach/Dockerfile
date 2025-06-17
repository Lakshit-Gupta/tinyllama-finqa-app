FROM public.ecr.aws/lambda/python:3.11

# Set environment variables at container level
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_home
ENV XDG_CACHE_HOME=/tmp/cache
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface_hub
ENV TORCH_HOME=/tmp/torch_home
ENV TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
ENV TMPDIR=/tmp
ENV TEMP=/tmp
ENV TMP=/tmp
ENV TOKENIZERS_PARALLELISM=false

# Copy requirements files first to leverage Docker caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install build dependencies
RUN yum install -y gcc make

# Create cache directories
RUN mkdir -p /tmp/transformers_cache /tmp/hf_home /tmp/cache /tmp/huggingface_hub /tmp/torch_home /tmp/torch_extensions

# Install PyTorch and dependencies in steps for better caching
# 1. Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Install NumPy separately to avoid build issues 
RUN pip install --no-cache-dir numpy==1.24.3

# 3. Install remaining dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application code separately (changes less frequently)
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Ensure correct permissions
RUN chmod 644 ${LAMBDA_TASK_ROOT}/lambda_handler.py && \
    chmod 755 ${LAMBDA_TASK_ROOT}/app

# Clean up to reduce image size
RUN yum remove -y gcc make && \
    yum clean all && \
    rm -rf /var/cache/yum

CMD [ "lambda_handler.lambda_handler" ]