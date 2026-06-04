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


def get_model_and_tokenizer():
    """
    Inizializza e restituisce il modello Transformer pre-addestrato e il relativo tokenizer.
    
    Scarica i pesi dal repository ufficiale Hugging Face configurato nella costante model.
    
    Returns:
        tuple: (AutoModelForSequenceClassification, AutoTokenizer) pronti per l'inferenza.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return model, tokenizer



def predict_sentiment_batch(batch, model, tokenizer):
    """
    Elabora un batch di testi estratti dal dataset ed esegue l'inferenza con RoBERTa.
    
    Gestisce la tokenizzazione con troncamento a 64 token per ottimizzare la memoria 
    e applica l'operatore argmax sui logit per estrarre la classe predetta.

    Args:
        batch (dict): Un sottoinsieme del dataset contenente la chiave "text".
        model: Il modello RoBERTa caricato.
        tokenizer: Il tokenizer associato al modello.

    Returns:
        dict: Un dizionario contenente la lista delle label predette ("predicted_label").
    """
    inputs = tokenizer(batch["text"], padding=True, truncation=True,
                       max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return {"predicted_label": predictions}



def run_sentiment_pipeline(dataset_split, num_examples=500):
    """
    Orchestra l'intera pipeline di analisi del sentiment su uno split di dataset.
    
    Seleziona un sottoinsieme casuale di dati per velocizzare l'elaborazione,
    inizializza i componenti del modello ed esegue la mappatura parallela delle predizioni.

    Args:
        dataset_split (datasets.Dataset): Lo split del dataset (es. train, validation, test).
        num_examples (int, optional): Numero di record da analizzare. Default a 500.

    Returns:
        datasets.Dataset: Il dataset di Hugging Face arricchito con la colonna 'predicted_label'.
    """

    # 1. Preparazione dati
    small_ds = dataset_split.shuffle(seed=42).select(range(num_examples))

    # 2. Setup Modello
    model, tokenizer = get_model_and_tokenizer()

    # 3. Inferenza
    results = small_ds.map(
        lambda x: predict_sentiment_batch(x, model, tokenizer),
        batched=True,
        batch_size=8
    )

    return results



def evaluate_results(results):
    """
    Calcola, stampa a video e restituisce le metriche di accuratezza e F1-Score.
    
    Genera un report di classificazione dettagliato dividendo le performance
    sulle tre macro-aree di sentiment aziendale: Negativo, Neutrale e Positivo.

    Args:
        results (datasets.Dataset): Il dataset elaborato contenente 'label' e 'predicted_label'.

    Returns:
        tuple: (accuracy_score, f1_score) come valori float.
    """

    y_true = results["label"]
    y_pred = results["predicted_label"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nAccuratezza: {acc:.4f} | F1-Score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Negativo', 'Neutrale', 'Positivo']))

    return acc, f1



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

    # Esegue la pipeline
    model, tokenizer = get_model_and_tokenizer()
    results = ds.map(lambda x: predict_sentiment_batch(x, model, tokenizer), batched=True)

    # Salva i risultati
    results.to_csv(output_path)
    print(f"Risultati salvati in: {output_path}")



# Monitoraggio
def test_reputation_system():
    # Test solo su 5 tweet per fare un controllo rapido di "salute"
    results = run_sentiment_pipeline(small_test_dataset, num_examples=5)
    acc, f1 = evaluate_results(results)

    assert acc >= 0  # Verifica che la funzione restituisca un numero valido


def log_performance(accuracy, f1):
    # Salva le performance in un file CSV per monitorare il calo nel tempo
    with open("monitoring_log.csv", "a") as f:
        f.write(f"{pd.Timestamp.now()},{accuracy},{f1}\n")
