from datasets import load_dataset


from Modello import run_sentiment_pipeline, evaluate_results


def test_pipeline_integration():
    """
    Test di Integrazione della pipeline di Sentiment Analysis.
    
    Verifica che il flusso completo (estrazione dati, tokenizzazione, inferenza 
    con il modello RoBERTa e calcolo delle metriche) funzioni senza sollevare eccezioni.
    
    Il test viene limitato volutamente a 10 esempi per garantire un'esecuzione rapida all'interno della pipeline CI/CD di GitHub Actions,
    evitando sovraccarichi computazionali.
    """
    # Caricamento del dataset pubblico TweetEval 
    dataset = load_dataset("tweet_eval", "sentiment")
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

    # Esecuzione rapida della pipeline su un micro-campionamento di controllo
    results = run_sentiment_pipeline(small_test_dataset, num_examples=10)
    acc, f1 = evaluate_results(results)

    # Controllo di integrità: l'accuratezza deve restituire un valore numerico valido
    assert acc >= 0
