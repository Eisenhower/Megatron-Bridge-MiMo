# Bailing (Ling) Examples

This directory contains example scripts for [Ling 2.0](https://github.com/inclusionAI/Ling-V2) MoE language models by inclusionAI.

Ling 2.0 uses a high-sparsity Mixture of Experts (MoE) architecture with sigmoid routing, QK-Norm, and Half RoPE.

| Model | HF ID | Architecture | Params | Active Params |
|---|---|---|---|---|
| Ling-flash-2.0 | `inclusionAI/Ling-flash-2.0` | MoE (256 experts, top-8) | 100B | 6.1B |
| Ling-flash-base-2.0 | `inclusionAI/Ling-flash-base-2.0` | MoE (256 experts, top-8) | 100B | 6.1B |
| Ling-mini-2.0 | `inclusionAI/Ling-mini-2.0` | MoE (256 experts, top-8) | 16B | 1.5B |
| Ling-mini-base-2.0 | `inclusionAI/Ling-mini-base-2.0` | MoE (256 experts, top-8) | 16B | 1.5B |

## Inference Sizing

The `Ling-flash-2.0` and `Ling-flash-base-2.0` checkpoints are 100B-parameter MoE models. They are not suitable as single-node 8-GPU 80GB inference examples with `TP=2 EP=4`: this configuration leaves effectively no memory headroom for activations, KV cache, or communication buffers after loading weights.

Use the smaller `Ling-mini-2.0` variant for the default single-node 8-GPU smoke example in [inference.sh](inference.sh). Run the 100B flash variants only on a larger-memory or multi-node setup and choose `TP`, `EP`, and process count for that environment.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable for the base directory. Default: `/workspace`.

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs

## Checkpoint Conversion

See [conversion.sh](conversion.sh) for checkpoint conversion examples.

The conversion script defaults to `Ling-flash-2.0`. Set `MODEL_NAME=Ling-mini-2.0` before running it if you want the imported/exported checkpoints to match the default single-node inference example.

### Import HF → Megatron

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model inclusionAI/Ling-flash-2.0 \
    --megatron-path ${WORKSPACE}/models/Ling-flash-2.0 \
    --trust-remote-code
```

### Export Megatron → HF

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model inclusionAI/Ling-flash-2.0 \
    --megatron-path ${WORKSPACE}/models/Ling-flash-2.0/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Ling-flash-2.0-hf-export \
    --trust-remote-code
```

### Round-trip Validation

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id inclusionAI/Ling-flash-2.0 \
    --megatron-load-path ${WORKSPACE}/models/Ling-flash-2.0/iter_0000000 \
    --tp 1 --ep 8 \
    --trust-remote-code
```

## Inference

See [inference.sh](inference.sh) for text generation with:
- Hugging Face checkpoint (`inclusionAI/Ling-mini-2.0` by default)
- Exported HF checkpoint (after conversion export)
- Imported Megatron checkpoint (commented out by default)

The imported Megatron checkpoint command is left commented out because that load
path can need extra temporary memory during checkpoint restore. Uncomment it for
targeted validation on a setup with enough memory headroom.

The default single-node parallelism for 8 GPUs is `--tp 2 --ep 4` with `Ling-mini-2.0`.
TP×PP×EP must equal `--nproc_per_node`.

Override the model and parallelism explicitly for other environments:

```bash
MODEL_NAME=Ling-mini-2.0 TP=2 EP=4 NPROC_PER_NODE=8 bash examples/models/bailing/inference.sh
```

> **Note**: `Ling-flash-2.0` with `--tp 2 --ep 4` is not expected to fit on a single 8-GPU 80GB node. `--tp 1 --ep 8` works for conversion round-trip but may cause issues during autoregressive inference with single-token batches (empty token dispatch to some EP ranks). Use a larger-memory or multi-node inference setup for the 100B flash variants.

> **Note**: All Ling 2.0 models use custom HuggingFace code, so `--trust-remote-code` is required for conversion and inference.
