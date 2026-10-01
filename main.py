from src.retriever import retrieve
from src.generator import generate
from src.generator import score_answer
from src.rag_sequence import rag_sequence_score, generate_candidates

question = "Who authored Harry Potter?"

results = retrieve(question, 3)

candidates = generate_candidates(question, results)

print("Question:")
print(question)

print("\nGenerated candidates:")

for candidate in candidates:
    print(f"- {candidate}")

print("\nRAG-Sequence scores:")

for candidate in candidates:

    score = rag_sequence_score(
        question,
        candidate,
        results,
    )

    print(f"{candidate}: {score:.6f}")
