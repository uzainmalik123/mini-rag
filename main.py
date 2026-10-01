from src.retriever import retrieve
from src.generator import generate

question = "First person to walk on the moon?"

results = retrieve(question, 3)

top_passage = results[0]["passage"]

print("Question:")
print(question)

print("\nRetrieved passage:")
print(top_passage)

answer = generate(
    question,
    top_passage,
)

print("\nGenerated answer:")
print(answer)
