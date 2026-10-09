"""BaseAgent: Qwen model loading + optional LoRA + generation/log-prob hooks."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.generation import batch_generate, compute_log_probs_for_tokens


class BaseAgent(nn.Module):
    """Wraps a causal LM (Qwen2.5 family), optionally LoRA-wrapped."""

    def __init__(self, model_path: str, device: str = "cuda", lora_config=None,
                 use_8bit: bool = False):
        super().__init__()
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            # prefer a pad token distinct from eos, so early-stopped generations
            # are distinguishable from truncated ones
            for candidate in ("<|endoftext|>", "<|pad|>"):
                if candidate in self.tokenizer.get_vocab():
                    self.tokenizer.pad_token = candidate
                    break
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        # decoder-only models need left padding for correct batch generation
        self.tokenizer.padding_side = "left"

        self.policy_model = self._load_base(model_path, use_8bit)
        if self.policy_model.config.pad_token_id is None:
            self.policy_model.config.pad_token_id = self.tokenizer.pad_token_id

        if lora_config is not None:
            self.policy_model = self._wrap_lora(self.policy_model, lora_config, use_8bit)

        self.policy_model.to(self.device)
        self.policy_model.eval()

    def _load_base(self, model_path: str, use_8bit: bool):
        kwargs = {}
        if use_8bit and self.device.type == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
            return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        return AutoModelForCausalLM.from_pretrained(model_path)

    @staticmethod
    def _wrap_lora(base_model, lora_config, use_8bit: bool):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if isinstance(lora_config, dict):
            lora_config = LoraConfig(task_type="CAUSAL_LM", **lora_config)
        if use_8bit:
            base_model = prepare_model_for_kbit_training(base_model)
        return get_peft_model(base_model, lora_config)

    def train(self, mode: bool = True):
        # Only the LoRA adapters get gradients; base weights stay frozen.
        self.policy_model.train(mode)
        return self

    def generate_responses(self, prompts, max_new_tokens=256, temperature=1.0,
                           max_prompt_tokens=512):
        return batch_generate(
            self.policy_model, self.tokenizer, prompts,
            max_new_tokens=max_new_tokens, temperature=temperature,
            device=str(self.device), max_prompt_tokens=max_prompt_tokens,
        )

    def get_policy_log_probs(self, full_ids, response_spans, use_amp=False):
        attention_mask = (full_ids != self.tokenizer.pad_token_id).to(full_ids.device).long()
        return compute_log_probs_for_tokens(
            self.policy_model, full_ids, response_spans,
            attention_mask=attention_mask, use_amp=use_amp,
            device=str(self.device),
        )
