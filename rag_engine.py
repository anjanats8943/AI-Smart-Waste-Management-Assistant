import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from knowledge_base import documents


# -----------------------------
# Embedding Model
# -----------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedding_tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

embedding_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL
)


def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    return torch.sum(
        token_embeddings * input_mask_expanded,
        1
    ) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )


def create_embedding(text):

    encoded_input = embedding_tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        model_output = embedding_model(
            **encoded_input
        )

    return mean_pooling(
        model_output,
        encoded_input["attention_mask"]
    )[0]


# -----------------------------
# Create Knowledge Embeddings
# -----------------------------

document_embeddings = []

for doc in documents:

    text = doc["title"] + " " + doc["content"]

    embedding = create_embedding(text)

    document_embeddings.append(embedding)


# -----------------------------
# RAG Retrieval
# -----------------------------

def retrieve_rag_information(query, top_k=2):

    query_embedding = create_embedding(query)

    similarities = []

    for doc_embedding in document_embeddings:

        similarity = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            doc_embedding.unsqueeze(0)
        )

        similarities.append(
            float(similarity.item())
        )

    top_indices = sorted(
        range(len(similarities)),
        key=lambda i: similarities[i],
        reverse=True
    )[:top_k]

    results = []

    for index in top_indices:

        results.append({
            "title": documents[index]["title"],
            "content": documents[index]["content"],
            "similarity": similarities[index]
        })

    return results


# -----------------------------
# Local LLM
# -----------------------------

LLM_MODEL = "google/flan-t5-small"

llm_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)


# -----------------------------
# RAG Answer Generation
# -----------------------------

def generate_rag_answer(query):

    retrieved_documents = retrieve_rag_information(
        query,
        top_k=2
    )

    context = "\n\n".join(
        [
            f"{doc['title']}: {doc['content']}"
            for doc in retrieved_documents
        ]
    )

    prompt = f"""
You are a Smart Waste Management Assistant.

Answer the user's question using ONLY the information
provided in the knowledge context below.

Knowledge Context:
{context}

User Question:
{query}

Give a short, clear and practical answer.
Do not invent information.
"""

    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=80
        )

    answer = llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "answer": answer,
        "sources": retrieved_documents
    }