"""
qwen2.py

Class definition for all LLMs derived from Qwen2ForCausalLM.
"""
from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    Qwen2PromptBuilder,
)

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
QWEN2_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "qwen2.5-1.5b": {
        "llm_family": "qwen2.5", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen2.5-1.5B"
    },
    "qwen2.5-1.5b-instruct": {
        "llm_family": "qwen2.5", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen2.5-1.5B-Instruct"
    },
    "qwen2.5-3b": {
        "llm_family": "qwen2.5", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen2.5-3B"
    },
    "qwen2.5-7b-instruct": {
        "llm_family": "qwen2.5", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen2.5-7B-Instruct"
    },
}
# fmt: on


class Qwen2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 4096,
        mount_path: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            mount_path=mount_path,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **QWEN2_MODELS[llm_backbone_id],
        )

        # [Special Case] Qwen-2.5 PAD Token Handling --> for clarity, we add an extra token, no need to resize the model embedding layer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<s>"]})
        self.tokenizer.bos_token = "<s>"

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Qwen2PromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
