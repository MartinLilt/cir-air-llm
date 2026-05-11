"""Profile load-vs-compute split in AirLLM MLX path.

Wraps the persister's load_model with a timer; everything else (compute) is
the residual. Reports cold first-pass and warm steady-state separately.

Usage: python scripts/profile_layers.py [model] [n_tokens] [compression]
       compression in {None, '4bit'}; pass '4bit' literally for quantized.
"""
import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "air_llm"))

import mlx.core as mx
from airllm.airllm_llama_mlx import AirLLMLlamaMlx
from airllm.persist import ModelPersister

MODEL = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
COMP = sys.argv[3] if len(sys.argv) > 3 else None
if COMP == 'None':
    COMP = None

print(f"=== model: {MODEL}, n_tokens={N}, compression={COMP!r} ===")

# build model first (so persister singleton is initialized)
model = AirLLMLlamaMlx(MODEL, show_memory_util=False, compression=COMP)

# wrap persister with timing
persister = ModelPersister.get_model_persister()
orig_load = persister.load_model
events = []  # (idx, name, dt)
counter = [0]
def timed_load(layer_name, path):
    t0 = time.perf_counter()
    r = orig_load(layer_name, path)
    dt = time.perf_counter() - t0
    events.append((counter[0], layer_name, dt))
    counter[0] += 1
    return r
persister.load_model = timed_load

input_tokens = model.tokenizer(["I like"], return_tensors="np",
    return_attention_mask=False, truncation=True, max_length=128, padding=False)

t0 = time.perf_counter()
out = model.generate(mx.array(input_tokens["input_ids"]), max_new_tokens=N,
                     use_cache=True, return_dict_in_generate=True)
total = time.perf_counter() - t0

# slice events into prompt-pass vs each gen-token pass.
# layout: each pass loads (embed + n_layers + norm + lm_head) = 1 + L + 1 + 1 = L+3
# first pass = prompt; subsequent passes = each gen token (N total)
n_layers_in_model = len(model.layer_names) - 3  # subtract embed/norm/lm_head
per_pass = n_layers_in_model + 3
n_passes = len(events) // per_pass
passes = [events[i*per_pass:(i+1)*per_pass] for i in range(n_passes)]

print(f"\ngen total: {total:.2f}s  output: {out!r}")
print(f"loads: {len(events)} events across {n_passes} passes ({per_pass} loads/pass)")

for pass_idx, ev in enumerate(passes):
    t_load = sum(dt for _, _, dt in ev)
    label = "prompt" if pass_idx == 0 else f"token {pass_idx}"
    pct = t_load / total * 100 if total else 0
    print(f"  pass {pass_idx:>2} ({label:>8}): t_load={t_load*1000:>6.1f} ms  ({pct:.1f}% of total)")

t_load_total = sum(dt for _, _, dt in events)
t_compute = total - t_load_total
print(f"\nsplit:")
print(f"  total:    {total*1000:>7.0f} ms")
print(f"  load:     {t_load_total*1000:>7.0f} ms  ({t_load_total/total*100:.1f}%)")
print(f"  compute:  {t_compute*1000:>7.0f} ms  ({t_compute/total*100:.1f}%)")

if n_passes >= 2:
    cold = sum(dt for _, _, dt in passes[0])
    warm_avg = statistics.mean(sum(dt for _, _, dt in p) for p in passes[1:])
    print(f"\ncold vs warm load:")
    print(f"  cold (prompt pass):  {cold*1000:>6.1f} ms")
    print(f"  warm (avg gen tok):  {warm_avg*1000:>6.1f} ms  ({(1-warm_avg/cold)*100:.0f}% faster)")
    print(f"  → page cache effect: {cold/warm_avg:.1f}x")