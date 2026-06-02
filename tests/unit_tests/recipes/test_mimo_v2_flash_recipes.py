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

#
# Test purpose:
# - Parametrize over the exported MiMo-V2-Flash recipe functions.
# - Monkeypatch AutoBridge so no HF download / 309B config is needed.
# - Build each config and assert it forms a valid ConfigContainer with the
#   MoE/attention kernel settings the recipe is supposed to apply.
#

import importlib
from typing import Callable

import pytest


_mimo_module = importlib.import_module("megatron.bridge.recipes.mimo_v2_flash.mimo_v2_flash")

_PRETRAIN_FUNCS = [_mimo_module.mimo_v2_flash_pretrain_config]
_SFT_FUNCS = [_mimo_module.mimo_v2_flash_sft_config]
_PEFT_FUNCS = [_mimo_module.mimo_v2_flash_peft_config]


class _FakeModelCfg:
    """Plain stand-in for the Megatron provider; accepts arbitrary recipe setattr."""

    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64

    def finalize(self):
        return None


class _FakeAutoBridge:
    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    for component in (
        cfg.model,
        cfg.train,
        cfg.optimizer,
        cfg.scheduler,
        cfg.dataset,
        cfg.logger,
        cfg.tokenizer,
        cfg.checkpoint,
        cfg.rng,
    ):
        assert component is not None
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1
    assert cfg.dataset.seq_length >= 1


def _assert_moe_kernels(cfg):
    # _apply_moe_kernels must have run for every recipe.
    assert cfg.model.transformer_impl == "transformer_engine"
    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.moe_permute_fusion is True
    # Force load balancing must stay off (RL/inference parity for noaux_tc routing).
    assert cfg.model.moe_router_force_load_balancing is False
    # MiMo-V2-Flash bridge does not support context parallelism yet.
    assert cfg.model.context_parallel_size == 1


@pytest.mark.parametrize("recipe_func", _PRETRAIN_FUNCS, ids=lambda f: f.__name__)
def test_pretrain_recipe(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_mimo_module, "AutoBridge", _FakeAutoBridge)
    cfg = recipe_func()
    _assert_basic_config(cfg)
    _assert_moe_kernels(cfg)
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.peft is None
    assert cfg.mixed_precision == "bf16_mixed"


@pytest.mark.parametrize("recipe_func", _SFT_FUNCS, ids=lambda f: f.__name__)
def test_sft_recipe(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_mimo_module, "AutoBridge", _FakeAutoBridge)
    cfg = recipe_func()
    _assert_basic_config(cfg)
    _assert_moe_kernels(cfg)
    assert cfg.peft is None
    assert cfg.mixed_precision == "bf16_mixed"


@pytest.mark.parametrize("recipe_func", _PEFT_FUNCS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_peft_recipe(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_mimo_module, "AutoBridge", _FakeAutoBridge)
    cfg = recipe_func(peft_scheme=peft_scheme)
    _assert_basic_config(cfg)
    _assert_moe_kernels(cfg)
    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")
