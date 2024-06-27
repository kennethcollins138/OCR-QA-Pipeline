FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y software-properties-common git wget poppler-utils && \
    apt-get install -y python3.10 python3-pip

RUN pip3 install nougat-ocr==0.1.0 transformers==4.25.1 requests==2.26.0 torch==1.9.0 pypdf==3.0.0 tqdm==4.62.3 pathlib==1.0.1

COPY . /app

CMD ["python3", "nougat_ocr.py"]