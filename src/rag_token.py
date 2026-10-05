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

    for step in range(max_new_tokens):

        probabilities = marginalize_next_token_probabilities(
            question,
            results,
            prefix_ids=prefix_ids,
        )

        next_token_id = probabilities.argmax(
            dim=-1,
            keepdim=True,
        )

        next_token = tokenizer.decode([next_token_id.item()])

        print(f"Step {step + 1}: " f"{next_token!r}")

        prefix_ids = torch.cat(
            [prefix_ids, next_token_id],
            dim=-1,
        )

        if next_token_id.item() == 2:
            break

    answer = tokenizer.decode(
        prefix_ids[0],
        skip_special_tokens=True,
    )

    return answer


def rag_token_beam_search(
    question,
    results,
    beam_size=2,
    max_new_tokens=20,
):
    """
    Generate an answer using RAG-Token
    with beam search.
    """

    start_token_id = torch.tensor(
        [[2]],
        device="cuda",
    )

    beams = [
        {
            "prefix_ids": start_token_id,
            "score": 0.0,
        }
    ]

    for step in range(max_new_tokens):

        candidates = []

        for beam in beams:

            prefix_ids = beam["prefix_ids"]
            current_score = beam["score"]

            # Get RAG-Token's combined next-token distribution.
            probabilities = marginalize_next_token_probabilities(
                question,
                results,
                prefix_ids=prefix_ids,
            )

            # Convert probabilities to log probabilities.
            log_probabilities = torch.log(probabilities.clamp(min=1e-12))

            # Take the best possible next tokens.
            top_log_probs, top_token_ids = log_probabilities.topk(beam_size)

            for log_prob, token_id in zip(
                top_log_probs[0],
                top_token_ids[0],
            ):

                new_prefix_ids = torch.cat(
                    [
                        prefix_ids,
                        token_id.view(1, 1),
                    ],
                    dim=-1,
                )

                new_score = current_score + log_prob.item()

                candidates.append(
                    {
                        "prefix_ids": new_prefix_ids,
                        "score": new_score,
                    }
                )

        # Keep the best beams.
        candidates.sort(
            key=lambda beam: beam["score"],
            reverse=True,
        )

        beams = candidates[:beam_size]

        print(f"\nStep {step + 1}")

        for index, beam in enumerate(beams):

            tokens = tokenizer.convert_ids_to_tokens(beam["prefix_ids"][0])

            print(f"Beam {index + 1}: " f"{tokens} " f"score={beam['score']:.6f}")

        # Stop if every beam has ended.
        if all(beam["prefix_ids"][0, -1].item() == 2 for beam in beams):
            break

    best_beam = beams[0]

    answer = tokenizer.decode(
        best_beam["prefix_ids"][0],
        skip_special_tokens=True,
    )

    return answer, best_beam["score"]
