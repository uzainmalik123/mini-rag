from src.retriever import retrieve
from src.rag_token import rag_token_generate

question = "Who discovered penicillin?"

results = retrieve(
    question,
    k=3,
)

answer = rag_token_generate(
    question,
    results,
)

print("Question:")
print(question)

print("\nRAG-Token answer:")
print(answer)
