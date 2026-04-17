FROM python:3.10-slim
WORKDIR /app
# Copia solo i file necessari, NON le cartelle env_monitoring!
COPY requirements.txt .
COPY app.py .
COPY FastText.py .
COPY README.md .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]