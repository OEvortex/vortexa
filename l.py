from vortexa.core.inference import VortexEmbedInference, similarity

# Load model using the "mini" model alias
model = VortexEmbedInference("mini")

queries = [
    "What is the capital of India?",
    "Explain gravity and general relativity",
]
documents = [
    "The capital of India is New Delhi.",
    "Gravity is a fundamental interaction that causes mutual attraction between all things with mass or energy.",
]

# 1. Encode queries and documents
query_embeddings = model.encode(queries)
document_embeddings = model.encode(documents)

# 2. Compute similarity matrix directly
similarity_matrix = query_embeddings @ document_embeddings.T
print("Similarity Matrix:")
print(similarity_matrix)
# Example output:
# [[0.82, 0.12],
#  [0.11, 0.74]]

# 3. Use built-in model similarity method
scores = model.similarity(query_embeddings, document_embeddings)

# 4. Single query against document list lookup
scores_single = model.similarity("What is the capital of India?", documents)
print("Single query scores:", scores_single)
