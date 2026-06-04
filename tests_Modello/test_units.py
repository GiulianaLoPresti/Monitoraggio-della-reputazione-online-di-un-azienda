from datasets import load_dataset
from Modello import run_sentiment_pipeline, evaluate_results
import time


"""
Test Unitari e di Validazione

Verifica la robustezza dei dizionari di mappatura delle etichette e la tenuta delle performance minime del modello Transformer.
"""
def map_label(label_id):
    mapping = {0: "Negativo", 1: "Neutrale", 2: "Positivo"}
    return mapping.get(label_id, "Sconosciuto")


def test_map_label_correct():
    assert map_label(0) == "Negativo"
    assert map_label(1) == "Neutrale"
    assert map_label(2) == "Positivo"


def test_map_label_invalid():
    # Verifica che un ID inesistente non faccia crashare il sistema
    assert map_label(99) == "Sconosciuto"


"""
    Test di integrazione delle performance.
    
    Verifica che il modello RoBERTa mantenga un livello di accuratezza minimo pari al 70% sul dataset di test reale 'tweet_eval'. 
    """
def test_reputation_system():
    # Caricamento e campionamento dei dati per il test di integrazione
    dataset = load_dataset("tweet_eval", "sentiment")
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
    # Esecuzione della pipeline di analisi
    results = run_sentiment_pipeline(small_test_dataset, num_examples=5)
    acc, f1 = evaluate_results(results)

    # Vincolo MLOps di accuratezza minima richiesta
    assert acc >= 0.70, f"Accuratezza troppo bassa: {acc}"



"""
    Test di performance non funzionale.
    
    Garantisce che il tempo di risposta per l'inferenza di un singolo tweet sia inferiore a 0.5 secondi. 
    Questo assicura la scalabilità dell'applicazione in un contesto di monitoraggio social in tempo reale.
    """

def test_inference_speed():
    
    # Monitoraggio del tempo di esecuzione dell'inferenza
    start_time = time.time()
    run_sentiment_pipeline(small_test_dataset, num_examples=1)
    duration = time.time() - start_time
    # Vincolo sui tempi di risposta aziendali
    assert duration < 0.5, f"Inference troppo lenta: {duration}s"
