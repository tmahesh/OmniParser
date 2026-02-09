FROM docker.io/pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install uv
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache -r requirements.txt

# RUN --mount=type=cache,target=/root/.cache/pip pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

#download models
RUN paddlex --pipeline OCR
RUN hf download microsoft/Florence-2-base

COPY . /app

EXPOSE 8080

CMD ["python", "gradio_demo.py"]
