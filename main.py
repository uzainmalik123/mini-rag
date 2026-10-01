from src.retriever import retrieve
from src.generator import generate
from src.generator import score_answer
from src.rag_sequence import rag_sequence_score

question = "Who discovered penicillin?"

candidates = [
    "Alexander Fleming",
    "Francis Crick",
]

print("Question:")
print(question)

print("\nCandidate scores:")

for answer in candidates:

    score = rag_sequence_score(
        question,
        answer,
        k=3,
    )

    print(f"{answer}: {score:.6f}")

results = retrieve(question, 3)

print("Question:")
print(question)

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
