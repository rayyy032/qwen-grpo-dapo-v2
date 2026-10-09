"""GRPOAgent: LoRA policy + frozen reference model for KL divergence."""

import torch
from transformers import AutoModelForCausalLM

from src.agents.base_agent import BaseAgent
from src.generation import compute_log_probs_for_tokens


class GRPOAgent(BaseAgent):
    """Policy (LoRA) + frozen reference (base, no LoRA by default).

    The reference is the frozen *base* model (without LoRA adapters) for two
    reasons: (1) it is the standard DAPO/GRPO KL anchor, (2) it halves memory
    by not loading a second adapter set.  Set ``ref_use_lora=True`` to instead
    freeze a LoRA-initialized copy.
    """

    def __init__(self, model_path: str, device: str = "cuda", lora_config=None,
                 use_8bit: bool = False, ref_use_lora: bool = False):
        super().__init__(model_path, device, lora_config, use_8bit)

        self.reference_model = AutoModelForCausalLM.from_pretrained(model_path)
        if ref_use_lora and lora_config is not None:
            self.reference_model = self._wrap_lora(self.reference_model,
                                                   lora_config, use_8bit)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.to(self.device)
        self.reference_model.eval()
        if self.reference_model.config.pad_token_id is None:
            self.reference_model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def get_reference_log_probs(self, full_ids, response_spans, use_amp=False):
        attention_mask = (full_ids != self.tokenizer.pad_token_id).to(full_ids.device).long()
        return compute_log_probs_for_tokens(
            self.reference_model, full_ids, response_spans,
            attention_mask=attention_mask, use_amp=use_amp,
            device=str(self.device),
        )
