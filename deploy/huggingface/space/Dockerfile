FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

# Build context: repository root. No model weights or authentication are baked in.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY deploy/huggingface/space/app.py ./app.py
RUN python -m pip install 'numpy==1.26.4' 'scipy==1.15.2' 'Pillow==11.1.0' \
    'PyYAML==6.0.2' 'transformers==4.57.3' 'safetensors==0.5.3' 'einops==0.8.1' \
    'numba==0.59.1' 'llvmlite==0.42.0' 'huggingface_hub>=0.36,<2' \
    'gradio>=5.49,<6' 'rembg==2.0.61' 'opencv-python-headless>=4.9,<4.12' 'onnxruntime>=1.18' 'trimesh>=4' \
    && python -m pip install --no-deps . \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface \
    U2NET_HOME=/home/appuser/.cache/rembg \
    MESHAXIOM_OUTPUT_DIR=/tmp/meshaxiom-outputs
EXPOSE 7860
CMD ["python", "app.py"]
