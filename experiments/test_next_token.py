import torch

from src.generator import (
    generator,
    tokenizer,
    get_next_token_probabilities,
)

question = "Who discovered penicillin?"
passage = "Alexander Fleming discovered penicillin in 1928."


# -------------------------
# Test 1: Empty prefix
# -------------------------

probabilities = get_next_token_probabilities(
    question,
    passage,
)

top_probabilities, top_tokens = probabilities.topk(5)

print("Empty prefix:")
for probability, token_id in zip(
    top_probabilities[0],
    top_tokens[0],
):
    token = tokenizer.decode([token_id])
    print(f"{token!r}: {probability.item():.4f}")


# -------------------------
# Test 2: Existing prefix
# -------------------------

prefix_ids = torch.tensor(
    [[2, 0, 33409]],
    device=generator.device,
)

probabilities = get_next_token_probabilities(
    question,
    passage,
    prefix_ids,
)

top_probabilities, top_tokens = probabilities.topk(5)

print("\nPrefix: </s> <s> Alexander")

for probability, token_id in zip(
    top_probabilities[0],
    top_tokens[0],
):
    token = tokenizer.decode([token_id])
    print(f"{token!r}: {probability.item():.4f}")
