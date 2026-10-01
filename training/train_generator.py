import torch

from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration

from data.qa_pairs import qa_pairs

MODEL_NAME = "facebook/bart-base"
SAVE_PATH = "model/bart-qa-mini"

# GPU if available else CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training on: ", device)

# Loading pretrained BART

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

model.to(device)
model.train()

# Build training inputs

inputs, answers = [], []

for example in qa_pairs:
    combined_input = (
        f"question: {example['question']} " f"context: {example['passage']}"
    )

    inputs.append(combined_input)
    answers.append(example["answer"])

# Encoding question + passage

encoded_inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

# Encoding answers

encoded_answers = tokenizer(
    text_target=answers, truncation=True, padding=True, return_tensors="pt"
)

input_ids = encoded_inputs["input_ids"].to(device)

attention_mask = encoded_inputs["attention_mask"].to(device)

labels = encoded_answers["input_ids"].to(device)


# Ignore padding when calculating loss

labels[labels == tokenizer.pad_token_id] = -100

# Optimizer

optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
)

# The actual training loop

epochs = 30

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} " f"- Loss: {loss.item():.4f}")

# Save the trained model

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\nSaved model to:", SAVE_PATH)
