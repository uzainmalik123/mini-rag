from src.retriever import retrieve
from src.rag_token import rag_token_generate, rag_token_beam_search

question = "Who discovered penicillin?"

results = retrieve(
    question,
    k=3,
)

# BEAM GENERATION

answer, score = rag_token_beam_search(
    question,
    results,
    beam_size=1,
)

print("\nBeam-search RAG-Token answer:")
print(answer)

print("\nBeam score:")
print(score)

# GREEDY GENERATION

# answer = rag_token_generate(
#     question,
#     results,
# )
#
# print("Question:")
# print(question)
#
# print("\nRAG-Token answer:")
# print(answer)
