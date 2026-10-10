# Small CPU-only image for the training Job.
FROM python:3.11-slim

# CPU wheel of PyTorch keeps the image under 1 GB.
RUN pip install --no-cache-dir torch numpy --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY train.py .

CMD ["python", "train.py"]
