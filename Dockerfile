# Usa un'immagine leggera di Python
FROM python:3.10-slim

# Imposta la cartella di lavoro nel container
WORKDIR /app

# Copia i file necessari nel container
COPY requirements.txt .
COPY app.py .
COPY FastText.py .
COPY conftest.py .
# Se hai altre cartelle necessarie per il codice (NON env_monitoring), copiale qui
# COPY Monitoraggio_reputazione/ ./Monitoraggio_reputazione/

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta usata da Streamlit
EXPOSE 7860

# Comando per avviare l'app (Hugging Face si aspetta la porta 7860)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]