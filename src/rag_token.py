import torch

from src.generator import get_next_token_probabilities, tokenizer


def marginalize_next_token_probabilities(
    question,
    results,
    prefix_ids=None,
):
    """
    Combine next-token probabilities across retrieved documents.

    P(y_t | x, y_<t) =
        sum_z P(z | x) * P(y_t | x, z, y_<t)
    """

    combined_probabilities = None

    for result in results:

        retrieval_probability = result["probability"]

        token_probabilities = get_next_token_probabilities(
            question,
            result["passage"],
            prefix_ids,
        )

        weighted_probabilities = retrieval_probability * token_probabilities

        if combined_probabilities is None:
            combined_probabilities = weighted_probabilities
        else:
            combined_probabilities += weighted_probabilities

    return combined_probabilities


def rag_token_generate(
    question,
    results,
    max_new_tokens=20,
):
    """
    Generate an answer one token at a time
    using RAG-Token marginalization.
    """

    # Start with BART's decoder start token.
    prefix_ids = torch.tensor(
        [[2]],
        device="cuda",
    )

    for _ in range(max_new_tokens):

        probabilities = marginalize_next_token_probabilities(
            question,
            results,
            prefix_ids=prefix_ids,
        )

        # Pick the most probable next token.
        next_token_id = probabilities.argmax(
            dim=-1,
            keepdim=True,
        )

        # Add the new token to our prefix.
        prefix_ids = torch.cat(
            [prefix_ids, next_token_id],
            dim=-1,
        )

        # Stop if we generated EOS.
        if next_token_id.item() == 2:
            break

    answer = tokenizer.decode(
        prefix_ids[0],
        skip_special_tokens=True,
    )

    return answer
