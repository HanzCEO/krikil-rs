# krikil-rs

High-performance, CPU-only Rust inference engine for [basically-ai/Pebble-25M-Chat](https://huggingface.co/basically-ai/Pebble-25M-Chat).

`krikil` runs the hybrid Mamba2 + Transformer architecture entirely on CPU in fp32 without quantization degradation, featuring:
- Exact Mamba2 state-space duality (SSD) and causal 1D depthwise convolution.
- Causal multi-head attention with Rotary Position Embeddings (RoPE) and key-value (KV) caching.
- Optimized AVX2 / FMA CPU tensor math kernels and GEMM acceleration.
- Embedded Jinja chat template engine with fallback.
- Streaming token output and sampling controls (temperature, top-p, top-k, repetition penalty).

## Installation & Build

Ensure Rust 1.80+ is installed:

```bash
cargo build --release
```

The optimized binary will be produced at `target/release/krikil`.

## Usage

```bash
krikil -m model.safetensors --template chat_template.jinja --context 2048 "What is the capital of France?"
```

If `--template` is omitted, the built-in Pebble chat template is used automatically:

```bash
krikil -m model.safetensors --context 2048 "What is the capital of France?"
```

### Options

- `-m, --model <PATH>`: Path to model safetensors weights (default: `model.safetensors`).
- `--tokenizer <PATH>`: Path to `tokenizer.json` (auto-detected next to model if omitted).
- `--template <PATH>`: Path to Jinja template file (defaults to built-in template).
- `--context <NUM>`: Context window limit (default: `2048`).
- `--max-tokens <NUM>`: Maximum new tokens to generate (default: `128`).
- `--temp <FLOAT>`: Temperature for sampling; set `0.0` for greedy decoding (default: `0.7`).
- `--top-p <FLOAT>`: Nucleus sampling cumulative threshold (default: `0.95`).
- `--top-k <NUM>`: Top-K filtering count (default: `50`).
- `--repetition-penalty <FLOAT>`: Repetition penalty (default: `1.2`).
- `--seed <NUM>`: Random seed for deterministic generation.
- `--quiet`: Suppress timing and throughput statistics.

## Testing & Verification

Run the test suite:

```bash
cargo test --release
```
