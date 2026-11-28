# Transformers (Part 6)

Folder for **[LLM Architectures Explained: Transformers (Part 6)](https://medium.com/@vipra_singh/llm-architectures-explained-understanding-transformers-part-6-3a5573ed30e7)**.

This article explains the Transformer architecture, inference and its optimizations without code listings, so there is no notebook here. The Transformer is built from scratch, with code, in the next part: [7_Coding_a_Transformer](../7_Coding_a_Transformer).

## Figures

The KV-cache animation in section 7.2 (*Key-Value (KV) Caching*) was made for the article and can be redrawn with Pillow:

```bash
pip install -r figures/requirements.txt
python figures/gif_kvcache.py out/kv_cache.gif
```

It also prints the cache size it illustrates: 0.5 MB per token for Llama 7B in 16-bit precision, so 2 GB for a 4,096-token context.
