FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tesseract-ocr wget && \
    pip install flask==2.1.2 pytesseract==0.3.8 requests==2.26.0 Pillow==9.0.1

COPY . /app

CMD ["python", "tesseract_ocr.py"]
