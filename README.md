# MLCoding

This repository contains a compact set of Python examples covering common machine learning, retrieval, and transformer-related patterns. Each script is intentionally small and self-contained so it can be used as a reference for core ideas in data processing and model logic.

## Repository contents

- [analyze_bug_sla.py](analyze_bug_sla.py) — Analyzes bug ticket data to compute resolution times, flag SLA breaches, and rank domains by breach severity.
- [context_compression.py](context_compression.py) — Builds a prompt context bundle by selecting retrieved chunks and prior conversation messages within a token budget.
- [cosine-similarity.py](cosine-similarity.py) — Computes cosine similarity between a query vector and a matrix of document vectors, returning the top-k matches.
- [event_stream_sessionization.py](event_stream_sessionization.py) — Segments a user event stream into sessions using time gaps and assigns session IDs, durations, and event counts.
- [masked_attention_calculation.py](masked_attention_calculation.py) — Implements a simplified self-attention mechanism with optional causal masking and a softmax helper.
- [reciprocal_rank_fusion.py](reciprocal_rank_fusion.py) — Fuses dense and sparse retrieval rankings using the reciprocal rank fusion (RRF) scoring formula.
- [sliding_window_chunking.py](sliding_window_chunking.py) — Splits text into overlapping sliding-window chunks for token-windowed processing.
- [softmax_with_temperature.py](softmax_with_temperature.py) — Applies temperature-scaled softmax to logits for probabilistic output control.

## Notes

These examples are designed for learning and experimentation rather than production deployment. They are useful for understanding how common ML and retrieval techniques are implemented in minimal Python form.
