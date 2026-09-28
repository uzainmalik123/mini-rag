import torch

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
)

model_name = "facebook/bart-large"

tokenizer = BartTokenizer.from_pretrained(model_name)

generator = BartForConditionalGeneration.from_pretrained(model_name)


def generate(question, passage):

    input_text = f"{question} {passage}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, maxLength=512)

    with torch.no_grad():
        output = generator.generate(**inputs, max_new_tokens=30)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer
