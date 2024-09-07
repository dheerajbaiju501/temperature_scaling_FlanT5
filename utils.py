def preprocess_batch(batch, tokenizer):
    """Preprocess a batch of data by tokenizing premise and hypothesis."""
    return tokenizer(batch['premise'], batch['hypothesis'], padding=True, truncation=True, return_tensors="pt")