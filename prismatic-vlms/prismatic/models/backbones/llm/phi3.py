"""
phi3.py

Class definition for all LLMs derived from Phi3ForCausalLM.
"""
from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import Phi3ForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
    Phi3PromptBuilder,
)

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
PHI3_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "phi3-3b": {
        "llm_family": "phi3", "llm_cls": Phi3ForCausalLM, "hf_hub_path": "Phi-3-mini-4k-instruct"
    },
    "phi3.5-3b": {
        "llm_family": "phi3", "llm_cls": Phi3ForCausalLM, "hf_hub_path": "Phi-3.5-mini-instruct"
    },
}
# fmt: on


class Phi3LLMBackbone(HFCausalLLMBackbone):
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
            **PHI3_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>"]})

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Phi3PromptBuilder
        # if self.identifier.startswith("llama2-") and self.identifier.endswith("-pure"):
        #     return PurePromptBuilder

        # elif self.identifier.startswith("llama2-") and self.identifier.endswith("-chat"):
        #     return LLaMa2ChatPromptBuilder

        # elif self.identifier.startswith("vicuna"):
        #     return VicunaV15ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Phi3DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Phi-3 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/phi3."""
        return torch.bfloat16
