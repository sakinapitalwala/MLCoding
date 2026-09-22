def build_prompt_context(system_prompt, conversation_history, retrieved_chunks, max_tokens):
    system_prompt_tokens = get_token_count(system_prompt)
    if system_prompt_tokens > max_tokens:
        raise ValueError("System prompt must not exceed max tokens")
    output_dict = {
        "system_prompt":system_prompt,
        "selected_chunks":[],
        "selected_messages":[],
        "total_tokens":0
    }   
 
    num_tokens = system_prompt_tokens
    for chunk in retrieved_chunks:
        chunk_tokens = get_token_count(chunk)
        if num_tokens + chunk_tokens > max_tokens:
            break
        else:
            output_dict['selected_chunks'].append(chunk)
            num_tokens = num_tokens + chunk_tokens
    
    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]["content"]
        msg_tokens = get_token_count(msg)
        if num_tokens + msg_tokens > max_tokens:            
            break
        else:
            output_dict['selected_messages'].append(conversation_history[i])
            num_tokens = num_tokens + msg_tokens
    output_dict['selected_messages'].reverse()
    output_dict["total_tokens"] = num_tokens
    return output_dict

def get_token_count(string):
    count = len(string.split())
    return count
if __name__ == "__main__":
    system_prompt = "You are an expert EDA tool assistant."  # 7 tokens

    conversation_history = [
        {"role": "user", "content": "How do I fix synthesis timing violation in block A?"},  # 10 tokens
        {"role": "assistant", "content": "Check the clock tree setup and path delay parameters."},  # 10 tokens
        {"role": "user", "content": "What parameter controls clock uncertainty?"}  # 6 tokens
    ]

    retrieved_chunks = [
        "Chunk 1: set_clock_uncertainty defines margin for setup and hold timing.",  # 10 tokens
        "Chunk 2: Use report_timing -delay_type max to inspect critical paths."    # 9 tokens
    ]

    max_tokens = 50
    print(build_prompt_context(system_prompt, conversation_history, retrieved_chunks, max_tokens))
