import math

from src.retriever import retrieve
from src.generator import score_answer


def rag_sequence_score(question, answer, k=3):

    results = retrieve(question, k)

    total_probability = 0.0

    for result in results:

        retrieval_probability = result["probability"]

        sequence_log_probability = score_answer(
            question,
            result["passage"],
            answer,
        )

        sequence_probability = math.exp(sequence_log_probability)

        contribution = retrieval_probability * sequence_probability

        total_probability += contribution

    return total_probability
