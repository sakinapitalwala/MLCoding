def scaled_dot_product_attention(Q, K, V, causal_mask=False):
    d_k = Q.shape[2]
    T = Q.shape[1]
    q_k_prod = (Q @ K.swapaxes(-1, -2))/np.sqrt(d_k)
    if causal_mask:
       lower_trian_mat = np.tril(np.ones((T,T)))
       q_k_prod_mask = np.where(lower_trian_mat == 0, -1e9, q_k_prod)
    else:
        q_k_prod_mask = q_k_prod
    softmax_qk = softmax(q_k_prod_mask, axis = -1)
    softmax_val = softmax_qk @ V
    return softmax_val, softmax_qk
    
def softmax(X, axis = -1):
    X_P = X - np.max(X, axis = axis, keepdims = True)
    sm = np.exp(X_P)/np.sum(np.exp(X_P), axis = axis, keepdims = True)
    return sm 


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    B, T, d_k, d_v = 2, 4, 8, 8
    Q = np.random.randn(B, T, d_k)
    K = np.random.randn(B, T, d_k)
    V = np.random.randn(B, T, d_v)

    # Test with causal_mask = True
    print(scaled_dot_product_attention(Q, K, V))
    print(scaled_dot_product_attention(Q, K, V, causal_mask = True))