import numpy as np
def temperature_softmax(logits, temperature = 1.0):
    if temperature <= 0:
        raise ValueError("temperaute cannot be less than or equal to 0")
    scaled_logits = logits-np.max(logits, axis = 1, keepdims = True)
    softmax = np.exp(scaled_logits/temperature)/np.sum(np.exp(scaled_logits/temperature), axis = 1, keepdims = True)
    return softmax

if __name__ == "__main__":
    logits = np.array([
    [1000.0, 1002.0, 1001.0],  # Large numbers (tests numerical stability)
    [2.0,    1.0,    0.0]      # Standard logits
    ])

    temperature = 1
    probs = temperature_softmax(logits, temperature)
    print(probs)
    print("Row Sums:", np.sum(probs, axis=1))