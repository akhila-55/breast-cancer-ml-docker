FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/akhila-55/breast-cancer-ml-docker"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY breast_cancer_model.pkl .
COPY metrics.json .

EXPOSE 5000

CMD ["python", "app.py"]