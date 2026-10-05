import numpy as np
def split_head(x, num_heads):
    d_model = x.shape[-1]
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    
    d_head = d_model // num_heads
    B = x.shape[0]
    T = x.shape[1]
    # X  -> (B, T, d_model)
    # X_reshaped -> (B, num_head, T, d_head)
    x_reshaped = x.reshape((B, T, num_heads, d_head))
    x_reshaped = x_reshaped.swapaxes(-3, -2)
    return x_reshaped

def merge_head(x):
    x = x.swapaxes(-3, -2)
    B, T, num_heads, d_head = x.shape
    x_reshaped = x.reshape(B, T, num_heads * d_head)
    return x_reshaped

if __name__ == "__main__":
    np.random.seed(42)

    B, T, d_model = 2, 4, 16
    num_heads = 4

    # Simulated linear projection output (B, T, d_model)
    x = np.random.randn(B, T, d_model)

    # 1. Test split_heads
    # Expected shape: (2, 4, 4, 4)

    # 2. Test merge_heads
    # Expected shape: (2, 4, 16)
    # Verification: np.allclose(x, merged_x) must be True!

    x_split = split_head(x, num_heads)
    print(x_split.shape)
    print(merge_head(x_split).shape)