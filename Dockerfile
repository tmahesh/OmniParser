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

#RUN pip install uv
#RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

#download models
#RUN paddlex --pipeline OCR
RUN hf download microsoft/Florence-2-base

RUN python - <<'PY'
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_angle_cls=False,
    use_gpu=False,
    show_log=False,
    max_batch_size=1024,
    use_dilation=True,
    det_db_score_mode="slow",
    rec_batch_num=1024,
)

img = np.zeros((64, 256, 3), dtype=np.uint8)
try:
    _ = ocr.ocr(img, cls=False)
except Exception:
    pass
print("PaddleOCR model cache prepared.")
PY

RUN python - <<'PY'
import numpy as np
import easyocr

reader = easyocr.Reader(["en"], gpu=False)
img = np.zeros((64, 256, 3), dtype=np.uint8)
try:
    _ = reader.readtext(img)
except Exception:
    pass
print("EasyOCR model cache prepared.")
PY

COPY . /app

EXPOSE 8080

CMD ["python", "gradio_demo.py"]
