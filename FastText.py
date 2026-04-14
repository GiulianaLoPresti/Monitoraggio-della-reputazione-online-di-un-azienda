from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
#from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix
import torch
from tqdm import tqdm
from datasets import load_dataset


# 1. Carica il dataset completo
dataset = load_dataset("tweet_eval", "sentiment")
# Etichette: 0 -> Negativo, 1 -> Neutrale, 2 -> Positivo
# Seleziona solo 500 esempi casuali per un test rapido
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(500))


print(f"Distribuzione classi nel campione: {dataset.unique('label')}")



MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# 2. Caricamento Dataset (usiamo il set di test per la valutazione)


# Mappa delle etichette: IMDb ha 0 (neg) e 1 (pos). 
# RoBERTa ha 0 (neg), 1 (neu), 2 (pos). 
# Dobbiamo allineare i risultati per il confronto.
label_map = {0: 0, 1: 2} # Mappa IMDb 0->0 e 1->2 (saltando il neutro)

def predict_sentiment(batch):
    # Tokenizzazione del batch
    inputs = tokenizer(batch["text"], padding=True, truncation=True, 
                       max_length=64, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Prendiamo la classe con il punteggio più alto
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return {"predicted_label": predictions}

# Usa questo nel .map() - uso un sottocampione del test set, per velocizzare il fit 
results = small_test_dataset.map(predict_sentiment, batched=True, batch_size=4)


# 3. Esecuzione dell'Inferenza in Batch (Veloce!)
# Regola batch_size in base alla tua memoria (8 o 16 per iniziare)
results = results.map(predict_sentiment, batched=True, batch_size=8)

# 4. Valutazione Qualitativa e Quantitativa
# Sostituisci queste righe:
# y_true = [label_map[l] for l in results["label"]]

# Con questa:
y_true = results["label"]
y_pred = results["predicted_label"]

# Generazione Report
print("\n--- REPORT DI VALUTAZIONE ---")
print(classification_report(y_true, y_pred, target_names=['Negativo', 'Neutrale', 'Positivo'], labels=[0, 1, 2]))

# Matrice di Confusione
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("Matrice di Confusione:")
print(cm)

# Se vedi una velocità di circa 1.5 - 3.0 it/s (iterazioni al secondo): Sta andando benissimo per essere una CPU. Finirai in circa 3 minuti.

from sklearn.metrics import accuracy_score, f1_score, classification_report

# ... (dopo aver ottenuto y_true e y_pred dal blocco precedente)

# 1. Calcolo dell'Accuratezza
accuracy = accuracy_score(y_true, y_pred)

# 2. Calcolo dell'F1-Score
# Usiamo 'weighted' perché tiene conto del numero di esempi per ogni classe
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n--- PERFORMANCE FINALI ---")
print(f"Accuratezza Totale: {accuracy:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")

# 3. Report dettagliato per singola classe
print("\n--- DETTAGLIO PER CLASSE ---")
report = classification_report(y_true, y_pred, 
                               target_names=['Negativo', 'Neutrale', 'Positivo'], 
                               labels=[0, 1, 2])
print(report)

# La categoria neutrale è a 0 perché IMDB ha solo classi positive o negative, ROberta invece fa previsione su 3 classi




def test_model():
# creo dei dati di esempio:
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    # Prima avevi 0.9, che era impossibile
    assert (accuracy >= 0.70) and (f1 > 0.70), f"Performance sotto soglia di accuracy ({accuracy}) e/o f1 score {f1} "  
    # Messaggio di errore
# Il test in questo caso consiste nel verificare che la metrica sia molto piccola: <1e-6

# Esecuzione del test, da terminale: pytest FastText.py