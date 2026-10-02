import torch

from src.retriever import retrieve
from src.rag_token import (
    marginalize_next_token_probabilities,
)
from src.generator import tokenizer

question = "Who discovered penicillin?"

results = retrieve(
    question,
    k=3,
)

prefix_ids = torch.tensor(
    [[2, 0, 33409]],
    device="cuda",
)

probabilities = marginalize_next_token_probabilities(
    question, results, prefix_ids=prefix_ids
)

top_probabilities, top_tokens = probabilities.topk(5)

print("Top 5 RAG-Token next tokens:")

for probability, token_id in zip(
    top_probabilities[0],
    top_tokens[0],
):
    token = tokenizer.decode([token_id])

    print(f"{token!r}: {probability.item():.4f}")
