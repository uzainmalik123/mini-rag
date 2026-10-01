import torch

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
)

model_name = "model/bart-qa-mini"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = BartForConditionalGeneration.from_pretrained(model_name)

generator.to(device)
generator.eval()


def generate(question, passage):

    input_text = f"question: {question} " f"context: {passage}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = generator.generate(
            **inputs,
            max_new_tokens=20,
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return answer
