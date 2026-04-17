from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
import pandas as pd

# MODEL TEST

# ---------- DOWNLOAD DATI ------------
# 1. Carica il dataset completo
# Scarico il subset "sentiment" del dataset TweetEval, contenente tweet reali
dataset = load_dataset("tweet_eval", "sentiment")
# Etichette: 0 -> Negativo, 1 -> Neutrale, 2 -> Positivo
# Seleziona solo 500 esempi casuali del test set per un test rapido
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
print(f"Distribuzione classi nel test set: {dataset['test'].unique('label')}")


# ---------- MODEL SETTING ------------
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# Carica e restituisce modello e tokenizer.
def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return model, tokenizer


# Funzione interna per processare i batch di testo
def predict_sentiment_batch(batch, model, tokenizer):
    inputs = tokenizer(batch["text"], padding=True, truncation=True,
                       max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return {"predicted_label": predictions}

# Modularizzazione principale:
# Prende un dataset, carica il modello, esegue le predizioni e restituisce i risultati.


def run_sentiment_pipeline(dataset_split, num_examples=500):

    # 1. Preparazione dati
    small_ds = dataset_split.shuffle(seed=42).select(range(num_examples))

    # 2. Setup Modello
    model, tokenizer = get_model_and_tokenizer()

    # 3. Inferenza
    # Usiamo una lambda per passare model e tokenizer alla funzione map
    results = small_ds.map(
        lambda x: predict_sentiment_batch(x, model, tokenizer),
        batched=True,
        batch_size=8
    )

    return results
# Usa questo nel .map() - uso un sottocampione del test set, per velocizzare il fit
# results = small_test_dataset.map(predict_sentiment, batched=True, batch_size=8)


# Calcola e stampa le metriche dai risultati ottenuti
def evaluate_results(results):

    y_true = results["label"]
    y_pred = results["predicted_label"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nAccuratezza: {acc:.4f} | F1-Score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Negativo', 'Neutrale', 'Positivo']))

    return acc, f1


# --- PUNTO DI INGRESSO (Per l'esecuzione normale) ---
if __name__ == "__main__":
    print("Avvio Pipeline di Monitoraggio...")
    full_dataset = load_dataset("tweet_eval", "sentiment")

    # Eseguiamo la pipeline modularizzata
    processed_results = run_sentiment_pipeline(full_dataset["test"])

    # Valutiamo
    evaluate_results(processed_results)


def load_and_predict(input_path, output_path):
    # Carica da CSV
    df = pd.read_csv(input_path)

    # Trasforma in formato Dataset di Hugging Face
    from datasets import Dataset
    ds = Dataset.from_pandas(df)

    # Esegue la pipeline (modificando leggermente run_sentiment_pipeline per accettare ds)
    model, tokenizer = get_model_and_tokenizer()
    results = ds.map(lambda x: predict_sentiment_batch(x, model, tokenizer), batched=True)

    # Salva i risultati
    results.to_csv(output_path)
    print(f"Risultati salvati in: {output_path}")

# Monitoraggio


def test_reputation_system():
    # Testiamo solo 5 tweet per fare un controllo rapido di "salute"
    results = run_sentiment_pipeline(small_test_dataset, num_examples=5)
    acc, f1 = evaluate_results(results)

    assert acc >= 0  # Verifichiamo che la funzione restituisca un numero valido


def log_performance(accuracy, f1):
    # Salva le performance in un file CSV per monitorare il calo nel tempo
    with open("monitoring_log.csv", "a") as f:
        f.write(f"{pd.Timestamp.now()},{accuracy},{f1}\n")


# Esecuzione del test, da terminale: pytest

# 1. Controlla lo stile e la forma
# flake8 .

# 2. Esegui i test unitari e di integrazione (mostrando i print con -s)
# pytest -s

# 3. (Opzionale) Formatta il codice automaticamente
# pip install black
# black .



# git add .github/workflows/main.yml
# git commit -m "Aggiunta pipeline CI/CD"
# git lfs install --skip-smudge
# git push origin main