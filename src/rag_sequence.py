import math

from src.retriever import retrieve
from src.generator import score_answer, generate


def rag_sequence_score(question, answer, results):

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


def generate_candidates(question, results):

    candidates = []

    for result in results:

        answer = generate(
            question,
            result["passage"],
        )

        candidates.append(answer)

    return candidates


def rag_sequence_generate(question, results):

    candidates = generate_candidates(
        question,
        results,
    )

    best_candidate = None
    best_score = float("-inf")

    for candidate in candidates:

        score = rag_sequence_score(
            question,
            candidate,
            results,
        )

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate, best_score
