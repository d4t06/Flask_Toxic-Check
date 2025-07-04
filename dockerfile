# stage 1
FROM python:3.9-slim-buster AS builder
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY index.py .
COPY viet_toxic_comment_model_no_transformers.pth .
COPY custom_vocab.json .

EXPOSE 5000
CMD ["python", "index.py"]