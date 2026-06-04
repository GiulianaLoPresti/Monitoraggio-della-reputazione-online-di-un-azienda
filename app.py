import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


st.title("Monitoraggio Reputazione Online")
st.write("Inserisci un tweet per analizzarne il sentiment.")

# Caricamento Modello 
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Input utente
user_input = st.text_area("Scrivi qui il tuo tweet:", "Oggi l'azienda ha fatto un ottimo lavoro!")

if st.button("Analizza"):
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=-1).item()

    # Mappa delle etichette
    labels = {0: "Negativo", 1: "Neutrale", 2: "Positivo"}
    result = labels[prediction]

    st.subheader(f"Risultato: {result}")


# Deploy del modello su Hugging Face
# Il deploy su Hugging Face rende il modello accesssibile tramite web garantendo la scalabilità della soluzione richiesta.

# La procedura prevede i seguenti passaggi:
# Validazione: Esecuzione dei test unitari e controllo dell'accuratezza minima (>= 70%).
# Autenticazione: Connessione sicura tramite HF_TOKEN memorizzato nei GitHub Secrets.
# Deploy Continuo (CD): Push automatico del codice e dei file di configurazione all'Hugging Face Space.
# Interfaccia Utente: Pubblicazione della web-app (tramite app.py) per l'analisi in tempo reale dei tweet aziendali.