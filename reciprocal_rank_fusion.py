import numpy as np
import pandas as pd

def reciprocal_rank_fusion(dense_df, sparse_df, k=60, top_n=5):
    dense_df = dense_df.sort_values(by = "dense_score", ascending = False)
    dense_df["dense_rank"] = [i for i in range(1,dense_df.shape[0] + 1)]
    sparse_df = sparse_df.sort_values(by = "bm25_score", ascending = False)
    sparse_df["sparse_rank"] = [i for i in range(1, sparse_df.shape[0] + 1)]
    combined_df = dense_df.merge(sparse_df, how = "outer", on = 'doc_id')
    combined_df["dense_rank"] = combined_df["dense_rank"].fillna(np.inf)
    combined_df["sparse_rank"] = combined_df["sparse_rank"].fillna(np.inf)
    
    # calculate RRF
    combined_df["rrf_score"] = 1/(k + combined_df["dense_rank"]) \
    + 1/(k + combined_df["sparse_rank"])
    
    output_df = combined_df.sort_values(by = "rrf_score", ascending = False)

    return output_df.iloc[:top_n]

if __name__ == "__main__":
    # Dense Vector Search Results (Ranked by Cosine Similarity)
    dense_data = {
        'doc_id': ['doc_A', 'doc_B', 'doc_C', 'doc_D'],
        'dense_score': [0.89, 0.82, 0.75, 0.61],
    }

    # Sparse BM25 Search Results (Ranked by BM25 Score)
    sparse_data = {
        'doc_id': ['doc_B', 'doc_E', 'doc_A', 'doc_F'],
        'bm25_score': [12.4, 10.1, 8.5, 4.2],
    }

    dense_df = pd.DataFrame(dense_data)
    sparse_df = pd.DataFrame(sparse_data)

    print(reciprocal_rank_fusion(dense_df, sparse_df, k=60, top_n=5))