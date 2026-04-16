from datasets import load_dataset
from FastText import run_sentiment_pipeline, evaluate_results

# Test su singole unità di codice
# deve poter girare senza che il codice in FastTest (file sorgente) venga eseguito

# TEST 1) mappatura delle label e assenza di label diverse dalle 3
# La logica che vuoi testare (puoi importarla dal tuo file principale)


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


# TEST 2)
def test_reputation_system():
    # Carichiamo i dati (pensa a questo come a un "mock" o un campione)
    dataset = load_dataset("tweet_eval", "sentiment")
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
    # Eseguiamo la pipeline modularizzata
    results = run_sentiment_pipeline(small_test_dataset, num_examples=5)
    acc, f1 = evaluate_results(results)

    # Questo è il cuore del test di sviluppo
    assert acc >= 0.70, f"Accuratezza troppo bassa: {acc}"
