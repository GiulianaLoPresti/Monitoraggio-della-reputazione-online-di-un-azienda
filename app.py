import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Titolo dell'app
st.title("Monitoraggio Reputazione Online")
st.write("Inserisci un tweet per analizzarne il sentiment.")

# Caricamento Modello (usiamo il caching per non ricaricarlo ogni volta)
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
    labels = {0: "Negativo 😡", 1: "Neutrale 😐", 2: "Positivo 😊"}
    result = labels[prediction]

    st.subheader(f"Risultato: {result}")