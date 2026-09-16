import numpy as np
def top_k_cosine_similarity(query_vec, doc_matrix, k=3):
    dot_product = np.dot(doc_matrix, query_vec)
    mag_query_vec = np.sqrt(np.sum(query_vec ** 2))
    mag_doc_matrix = np.sqrt(np.sum(doc_matrix ** 2, axis = 1))
    denominator = mag_query_vec * mag_doc_matrix
    denominator = np.where(denominator == 0, 1e-8, denominator)
    cosine_sim = dot_product / denominator
    print(cosine_sim)
    cosine_sim_ind = np.argsort(cosine_sim)[::-1][:k]
    cosine_sim_top = cosine_sim[cosine_sim_ind]
    return cosine_sim_ind, cosine_sim_top

if __name__ == "__main__":
    query_vec = np.array([1.0, 0.0, 1.0, 0.0])
    doc_matrix = np.array([
    [1.0, 0.0, 1.0, 0.0],  # Doc 0: Exact match -> Similarity = 1.0
    [1.0, 0.0, 0.0, 0.0],  # Doc 1: Partial match -> Similarity ≈ 0.7071
    [0.0, 1.0, 0.0, 1.0],  # Doc 2: Orthogonal (no match) -> Similarity = 0.0
    [1.0, 1.0, 1.0, 0.0],  # Doc 3: Partial match -> Similarity ≈ 0.8165
    [0.0, 0.0, 0.0, 0.0]   # Doc 4: Zero vector edge case -> Similarity = 0.0
    ])
    print(top_k_cosine_similarity(query_vec, doc_matrix, k=3))

