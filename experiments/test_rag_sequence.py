from data.qa_pairs import qa_pairs
from src.retriever import retrieve
from src.rag_sequence import rag_sequence_generate

for example in qa_pairs:

    question = example["question"]
    expected = example["answer"]

    results = retrieve(
        question,
        k=3,
    )

    answer, score = rag_sequence_generate(
        question,
        results,
    )

    print("\nQuestion:")
    print(question)

    print("Expected:")
    print(expected)

    print("Generated:")
    print(answer)

    print("Score:")
    print(f"{score:.6f}")

    print("-" * 50)
