# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo bundles several related subprojects under one tree. They share a venv and history but are otherwise independent — changes in one usually don't affect the others.

- `air_llm/` — the **AirLLM** pip package (layer-by-layer inference for large LLMs on small GPUs). Has its own `setup.py`, `README.md`, `tests/`, `examples/`.
- `training/` — Anima 33B QLoRA fine-tuning (`qlora.py` driven by `run_Amina_training.sh` / `run_finetune_raining_based_on_Anima.sh`).
- `rlhf/` — Anima DPO/RLHF training (`qlora_dpo.py` driven by `run_dpo_training.sh`).
- `anima_100k/` — long-context (100k) training: `longer_training.py` + `modeling_flash_llama.py`, driven by `run_longer_training.sh`. Includes notebooks for generating long-context retrieval eval datasets.
- `eval/` — Elo tournament eval notebook over translated Vicuna.
- `data/` — translated Vicuna eval set (+ notebook that produced it via GPT-4).
- `scripts/test_cn_dataset_lenghts.py` — token-length distribution check used to pick `source_max_len` / `target_max_len` for training scripts.
- Top-level `requirements.txt` is for the **training** subprojects (bitsandbytes, peft, accelerate, etc.), **not** for AirLLM. AirLLM's deps live in `air_llm/setup.py`.

## Common commands

### AirLLM package (`air_llm/`)

```bash
# Install for development
cd air_llm && pip install -e .

# Run all tests (unittest, run from repo root, NOT from air_llm/)
python -m unittest air_llm.tests.test_automodel
python -m unittest air_llm.tests.test_compression

# Single test
python -m unittest air_llm.tests.test_automodel.TestAutoModel.test_auto_model_should_return_correct_model

# Smoke-test inference
python air_llm/inference_example.py
```

Note: `test_automodel.py` uses a relative import (`from ..airllm.auto_model import AutoModel`), so it must be run as a module from the repo root, not as a script.

### Anima training subprojects

Each training flavor has a shell wrapper that calls the underlying Python entrypoint with the canonical hyperparameters (mostly from the QLoRA paper Appendix B, Table 9). Prefer running the shell script over invoking Python directly:

```bash
# QLoRA SFT (Anima 33B)
cd training && bash run_Amina_training.sh

# DPO RLHF
cd rlhf && bash run_dpo_training.sh

# 100k long-context training
cd anima_100k && bash run_longer_training.sh
```

Outputs land in `./Anima_run/output_<unix-timestamp>/` relative to the script's CWD. `wandb` is enabled by default (`--report_to 'wandb'`) — make sure `WANDB_API_KEY` is set or pass `--report_to none`.

## AirLLM architecture

AirLLM's central trick: instead of holding the whole model in VRAM, it splits the checkpoint into per-layer shards on disk and streams them through the GPU one at a time during the forward pass. That's why a 70B model fits on a 4 GB card.

Key pieces:

- **`AirLLMBaseModel` (`airllm_base.py`)** — the actual sharded forward pass and generation loop. Subclasses for each architecture only override `set_layer_names_dict()` (paths into the model graph: `embed`, `layer_prefix`, `norm`, `lm_head`) and any architecture-specific tokenizer/config quirks. When adding a new model family, this is almost always all that's needed.

- **`AutoModel.from_pretrained` (`auto_model.py`)** — dispatcher. Reads `config.architectures[0]` from HuggingFace and picks the right subclass by string match (`Qwen2ForCausalLM`, `Baichuan`, `ChatGLM`, `InternLM`, `Mistral`, `Mixtral`, `Llama`, falling back to Llama2). When adding a new architecture, register it here.

- **macOS vs Linux split** — both `airllm/__init__.py` and `auto_model.py` branch on `platform == "darwin"` at import time. On macOS only `AirLLMLlamaMlx` is imported (uses Apple's MLX framework); the CUDA-based subclasses are never loaded. `AutoModel.from_pretrained` ignores architecture detection on macOS and always returns the MLX backend. **Implication:** any change touching the import surface or `AutoModel` dispatch must work on both branches — test on Linux and macOS, or at least confirm import succeeds in both.

- **`persist/` package** — abstracts layer-shard storage. `safetensor_model_persister.py` (Linux/CUDA path) and `mlx_model_persister.py` (macOS path) implement the same interface defined by `model_persister.py`. They handle the disk format AirLLM creates the first time you load a model (saved under the HF cache as `splitted_model/` unless `layer_shards_saving_path` is passed).

- **Compression (`utils.py: compress_layer_state_dict` / `uncompress_layer_state_dict`)** — block-wise quantization (4bit/8bit via bitsandbytes) applied to the *saved shards*, not at runtime. Bottleneck is disk loading, so quantizing only the weights is enough — activations stay full precision. `bitsandbytes` is an optional import; the code falls back gracefully if not installed.

- **First-run behavior** — `AirLLMBaseModel.__init__` calls `find_or_create_local_splitted_path`, which downloads the HF model and converts it to the per-layer format on first use. This is disk-heavy. The `delete_original=True` flag drops the original HF download afterward to save ~50% disk.

## Things to know before editing

- The training subprojects (`training/`, `rlhf/`, `anima_100k/`) and AirLLM (`air_llm/`) are largely independent. Changes to one rarely need to touch the others.
- AirLLM has no linter or formatter config checked in — match surrounding style.
- The macOS code path is import-time gated. Don't move imports to module top level without checking that the platform branch still holds.
- Tests don't cover the actual sharded forward pass (it requires GPUs and large model downloads). `test_automodel` covers the dispatcher; `test_compression` covers the quantization round-trip. Treat these as smoke tests — manual notebook runs in `air_llm/examples/` are the real integration check.