# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training recipes for MiMo-V2-Flash (Xiaomi, 309B hybrid-attention MoE).

These recipes target BF16 training (the open weights ship FP8 and are dequantized to BF16
on import — see the bridge). FP8 *training* is intentionally out of scope.

The model is large (256 experts, 48 layers), so the defaults assume a multi-node setup;
adjust the parallelism fields for your cluster. RL training through verl drives the model
via ``AutoBridge`` directly and does not require these recipes.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


# Default HuggingFace checkpoint (BF16 dequantized weights recommended for training).
HF_MODEL_ID = "XiaomiMiMo/MiMo-V2-Flash"


def _apply_moe_kernels(cfg: ConfigContainer) -> ConfigContainer:
    """Common MoE/attention kernel and dispatcher settings for MiMo-V2-Flash."""
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    # noaux_tc routing config is set on the provider by the bridge; recipes must not
    # silently override it. Force-load-balancing must stay off (RL/inference parity).
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cuda_graph_impl = "none"
    return cfg


def mimo_v2_flash_pretrain_config() -> ConfigContainer:
    """Pre-training config for MiMo-V2-Flash.

    Recommended starting parallelism (multi-node): TP=4, PP=4, EP=8. Tune for your cluster.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(HF_MODEL_ID).to_megatron_provider(load_weights=False)

    # Tokenizer - NullTokenizer for mock-data pretraining smoke
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.context_parallel_size = 1  # MiMo-V2-Flash bridge does not support CP yet
    cfg.model.seq_length = 4096

    cfg.train.train_iters = 1000000
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.mixed_precision = "bf16_mixed"
    _apply_moe_kernels(cfg)
    return cfg


def mimo_v2_flash_sft_config() -> ConfigContainer:
    """Full SFT config for MiMo-V2-Flash.

    Recommended starting parallelism (multi-node): TP=4, PP=4, EP=8.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(HF_MODEL_ID).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = HF_MODEL_ID

    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.context_parallel_size = 1

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 50
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 5e-6

    cfg.mixed_precision = "bf16_mixed"
    _apply_moe_kernels(cfg)

    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    return cfg


def mimo_v2_flash_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """PEFT (LoRA/DoRA) config for MiMo-V2-Flash.

    Args:
        peft_scheme: "lora", "dora", or a custom :class:`PEFT` instance.
    """
    cfg = _peft_common()

    cfg.model = AutoBridge.from_hf_pretrained(HF_MODEL_ID).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = HF_MODEL_ID

    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length

    cfg.peft = default_peft_config(peft_scheme)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.context_parallel_size = 1

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 50
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    cfg.mixed_precision = "bf16_mixed"
    _apply_moe_kernels(cfg)

    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    return cfg
