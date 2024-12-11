from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

def compute_code_similarity(code1, code2):
    # Encode both code samples
    emb1 = model.encode(code1, convert_to_tensor=True)
    emb2 = model.encode(code2, convert_to_tensor=True)
    
    # Compute cosine similarity between the embeddings
    similarity = util.pytorch_cos_sim(emb1, emb2)
    return float(similarity[0][0])  # Convert from tensor to float

# Example usage
code1 = """def hello():
    print("Hello World")"""
code2 = """def greet():
    print("Hello World")"""

similarity = compute_code_similarity(code1, code2)
print(f"Similarity score: {similarity:.4f}")
