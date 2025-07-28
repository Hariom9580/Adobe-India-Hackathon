FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Challenge_1a_RF/bert_pdf_processor_final.py .

CMD ["python", "bert_pdf_processor_final.py"] 