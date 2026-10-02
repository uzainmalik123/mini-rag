import torch

from src.generator import generator, tokenizer

question = "Who discovered penicillin?"
passage = "Alexander Fleming discovered penicillin in 1928."

input_text = f"question: {question} context: {passage}"

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

inputs = {key: value.to(generator.device) for key, value in inputs.items()}


# Let BART generate normally first
with torch.no_grad():
    generated = generator.generate(
        **inputs,
        max_new_tokens=5,
    )

print("Full generated IDs:")
print(generated)

print("\nFull generated tokens:")
print(tokenizer.convert_ids_to_tokens(generated[0]))


# Take the prefix:
# </s> <s> Alexander
prefix_ids = generated[:, :3]

print("\nPrefix IDs:")
print(prefix_ids)

print("\nPrefix tokens:")
print(tokenizer.convert_ids_to_tokens(prefix_ids[0]))


# Ask BART what should come next
with torch.no_grad():
    outputs = generator(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=prefix_ids,
    )


logits = outputs.logits

next_token_logits = logits[:, -1, :]

probabilities = torch.softmax(
    next_token_logits,
    dim=-1,
)

top_probabilities, top_tokens = probabilities.topk(5)


print("\nTop 5 next tokens:")

for probability, token_id in zip(
    top_probabilities[0],
    top_tokens[0],
):
    token = tokenizer.decode([token_id])
    print(f"{token!r}: {probability.item():.4f}")
