from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

def compute_code_similarity(code1, code2):
    # Encode both code samples
    emb1 = model.encode(code1, convert_to_tensor=True)
    emb2 = model.encode(code2, convert_to_tensor=True)
    
    # Compute cosine similarity between the embeddings
    # Larger values indicate higher similarity - on a scale of -1 to 1
    similarity = util.pytorch_cos_sim(emb1, emb2)
    return float(similarity[0][0])  # Convert from tensor to float

