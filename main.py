from src.retriever import retrieve
from src.rag_sequence import rag_sequence_generate

question = "Who authored Harry Potter?"

results = retrieve(question, 3)

answer, score = rag_sequence_generate(
    question,
    results,
)

print("Question:")
print(question)

print("\nFinal answer:")
print(answer)

print("\nScore:")
print(f"{score:.6f}")
