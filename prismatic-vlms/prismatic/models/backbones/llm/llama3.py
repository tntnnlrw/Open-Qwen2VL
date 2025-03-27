"""
llama2.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""
from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, LLaMa3ChatPromptBuilder

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
LLAMA3_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "llama3.1-8b-pure": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "Llama-3.1-8B"
    },

    "llama3.1-8b-instruct": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "Llama-3.1-8B-Instruct"
    },
    
    "llama3.2-3b-pure": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "Llama-3.2-3B"
    },

    "llama3.2-3b-instruct": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "Llama-3.2-3B-Instruct"
    },
}
# fmt: on


class LLaMa3LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 8192,
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
            **LLAMA3_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        # Weizhi for new project: we did not need the added padding token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<|pad|>"]})
        self.tokenizer.pad_token = "<|pad|>"
        self.llm.resize_token_embeddings(len(self.tokenizer))

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return LLaMa3ChatPromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16
