def sliding_window_chunker(text, window_size, overlap_size):
    '''Splits text into overlapping chunks using a sliding 
    window approach.'''
    chunks = []
    tokens = text.split()
    #print(tokens)
    chunk = []
    count = 0
    i = 0
    if overlap_size >= window_size:
        raise ValueError("Overlap size must be strictly smaller than window size.")
    if not tokens:
        return []
    if len(tokens) < window_size:
        return [text.strip()]
    while i < len(tokens):
        if count < window_size:
            chunk.append(tokens[i])
            #print(chunk)
            count = count + 1
            i = i + 1
        else:
            chunks.append(' '.join(chunk))
            count = 0
            chunk = []
            i = i - overlap_size
    if count > 0:
            chunks.append(' '.join(chunk))
    return chunks

if __name__ == "__main__":
    sample_text = (
    "EDA tool synthesis engine handles timing constraints "
    "power optimization and clock tree synthesis across "
    "multiple hardware blocks"
    )

    window_size = 6
    overlap_size = 2
    print(sliding_window_chunker(sample_text, window_size, overlap_size))
    