from src.retriever import retrieve
from src.generator import generate

question = "Who authored Harry Potter?"

results = retrieve(question, 3)

print("Question:")
print(question)

for result in results:
    print(f"{result['rank']}. {result['passage']}")

passages = [result["passage"] for result in results]

context = " ".join(passages)

print("\nRetrieved passage:")
print(passages)

answer = generate(
    question,
    passages,
)

print("\nGenerated answer:")
print(answer)
