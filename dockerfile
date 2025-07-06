
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY index.py .
COPY model.pth .
COPY custom_vocab.json .

#FROM python:3.9-slim-buster
#WORKDIR /app
#COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
#COPY --from=builder /app /app

EXPOSE 5000

CMD ["python", "index.py"]