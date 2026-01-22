#!/usr/bin/env python3
"""
Minimal reproduction of the forward_jit.reset() issue.

The issue: Without reset(), the second generate() call only outputs 1 token.
Root cause: Graph caching issue - the graph caches input buffers internally
and doesn't update when different buffers are passed in subsequent calls.

Run with JIT=2 (no graph) to see it work correctly:
  JIT=2 python repro_jit_issue.py
"""
import sys
from tinygrad import Tensor, UOp, getenv
from tinygrad.apps.llm import Transformer, SimpleTokenizer, models

def generate_no_reset(model, tokens: list[int], start_pos=0):
  """generate() without the reset() call"""
  v_start_pos = UOp.variable("start_pos", 1, model.max_context-1)
  t = Tensor([tokens[start_pos:]], dtype="int32")
  # NO reset here - this is what we're testing
  while len(tokens) < model.max_context:
    t = model(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
    next_id = int(t.item())
    tokens.append(next_id)
    start_pos = len(tokens) - 1
    yield next_id

if __name__ == "__main__":
  model, kv = Transformer.from_gguf(Tensor.from_url(models["llama3.2:1b"]), max_context=4096)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id = kv['tokenizer.ggml.eos_token_id']

  def chat(prompt: str, ids: list[int], use_reset=True):
    start_pos = max(len(ids) - 1, 0)
    ids = ids + tok.role("user") + tok.encode(prompt) + tok.end_turn(eos_id) + tok.role("assistant")
    print(f"User: {prompt}\nAssistant: ", end="", flush=True)
    gen = model.generate(ids, start_pos) if use_reset else generate_no_reset(model, ids, start_pos)
    cnt = 0
    for next_id in gen:
      if next_id == eos_id: break
      cnt += 1
      sys.stdout.write(tok.decode([next_id]))
      sys.stdout.flush()
      if cnt >= 20: break
    print(f"\n[{cnt} tokens]\n")
    return ids

  ids = [bos_id] if bos_id is not None else []

  # First call with reset (initializes JIT properly)
  model.forward_jit.reset()
  ids = chat("Hello", ids, use_reset=True)

  # Second call WITHOUT reset - fails (only 1 token) when graphing is enabled
  # Works with JIT=2 (no graph)
  print("--- Second call without reset ---")
  ids = chat("Bye", ids, use_reset=False)
