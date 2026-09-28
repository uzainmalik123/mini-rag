import torch
import faiss

from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

# KNOWLEDGE BASE

from data.passages import passages

# LOAD BERTd

model_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_model_name = "facebook/dpr-question_encoder-single-nq-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
document_encoder = DPRContextEncoder.from_pretrained(model_name)

print("Document encoder loaded")

# ENCODE DOCUMENTS

inputs = tokenizer(passages, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    document_embeddings = document_encoder(**inputs).pooler_output

print(document_embeddings.shape)

document_vectors = document_embeddings.cpu().numpy().astype("float32")

# BUILD FAISS INDEX

dimension = document_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(document_vectors)

print("Documents in index: ", index.ntotal)


def retrieve(question, k):

    # LOAD BERTq

    question_tokenizer = AutoTokenizer.from_pretrained(question_model_name)
    question_encoder = DPRQuestionEncoder.from_pretrained(question_model_name)

    # ENCODE QUESTION

    question_inputs = question_tokenizer(question, return_tensors="pt")

    with torch.no_grad():
        question_embeddings = question_encoder(**question_inputs).pooler_output

    question_vector = question_embeddings.cpu().numpy().astype("float32")

    # MIPS / tok-k SEARCH

    scores, indices = index.search(question_vector, k)

    # RESULTS
    results = []

    for rank, document_index in enumerate(indices[0]):
        results.append(
            {
                "rank": rank + 1,
                "score": float(scores[0][rank]),
                "passage": passages[document_index],
            }
        )

    return results
