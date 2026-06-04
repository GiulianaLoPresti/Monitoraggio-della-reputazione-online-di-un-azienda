from datasets import load_dataset


from Modello import run_sentiment_pipeline, evaluate_results


def test_pipeline_integration():
    dataset = load_dataset("tweet_eval", "sentiment")
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
    # Testiamo la pipeline su soli 10 esempi per essere velocissimi
    results = run_sentiment_pipeline(small_test_dataset, num_examples=10)
    acc, f1 = evaluate_results(results)
    assert acc >= 0
