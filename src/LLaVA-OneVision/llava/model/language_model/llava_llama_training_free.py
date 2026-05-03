"""LLaVA-1.5 (LLaMA / Vicuna) training-free wrapper for VFlowOpt-style
LLM-side progressive image-token pruning.

This is the LLaMA counterpart of `llava_qwen_training_free.py`. It does NOT
implement the ViT-side iLLaVA changes (entropy-augmented attention reduction
in the vision tower); only the LLM-side progressive pruning at
``illava_llm_k`` layers is enabled. Token importance is scored from the LLM's
own self-attention at the layer immediately preceding each prune layer.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# Default to 576 image tokens per image for LLaVA-1.5 (CLIP-ViT-L/14-336).
DEFAULT_LLAVA15_IMAGE_TOKENS_PER_IMAGE = 576


class LlavaLlamaTrainingFreeConfig(LlamaConfig):
    model_type = "llava_llama_training_free"
    temperature: float = 0.0
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None


class LlavaLlamaTrainingFreeModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaTrainingFreeConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaTrainingFreeModel, self).__init__(config)


class LlavaLlamaTrainingFreeForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaLlamaTrainingFreeConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        config.model_type = "llava_llama_training_free"
        self.model = LlavaLlamaTrainingFreeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    @staticmethod
    def _compute_image_token_length(input_ids: torch.Tensor, image_features) -> int:
        """Return total number of image-feature tokens that will be inserted."""
        if isinstance(image_features, list):
            return sum(int(f.shape[0]) for f in image_features)
        # Tensor shape [num_images, tokens, hidden]
        return int(image_features.shape[0] * image_features.shape[1])

    @staticmethod
    def _find_image_start_index(input_ids: torch.Tensor) -> int:
        """First IMAGE_TOKEN_INDEX position in the (batch=1) input."""
        from llava.constants import IMAGE_TOKEN_INDEX

        ids = input_ids[0]
        positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        if positions.numel() == 0:
            return 0
        return int(positions[0].item())

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        illava_config=None,
        image_token_length=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes
            )

        if illava_config is not None and illava_config.get("enable_illava_llm", False):
            return super().forward_illava(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                illava_config=illava_config,
                image_token_length=image_token_length,
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        illava_config=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        image_start_index = 0
        image_token_length = 0

        if images is not None:
            image_start_index = self._find_image_start_index(inputs)
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )
            # image_token_length = total inserted image tokens; for LLaVA-1.5 each
            # image contributes 576 tokens.
            num_images = (
                len(images) if isinstance(images, list) else (images.shape[0] if images.dim() == 4 else 1)
            )
            image_token_length = DEFAULT_LLAVA15_IMAGE_TOKENS_PER_IMAGE * num_images
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if illava_config is not None and illava_config.get("enable_illava_llm", False) and image_token_length > 0:
            # Per-sample dynamic illava_llm_r computation, given total-token-basis
            # keep_ratio: ensure end-of-prefill image-token count matches
            #     n_keep_img = n_img - (1 - keep_ratio) * total_prompt_len
            # Then split the per-stage ratios so that the cumulative product
            # over stages equals (n_keep_img / n_img).
            total_len = int(inputs_embeds.shape[1])
            keep_ratio_total = float(illava_config.get("keep_ratio_total", 1.0))
            n_img = int(image_token_length)
            n_keep_img = int(round(n_img - (1.0 - keep_ratio_total) * total_len))
            n_keep_img = max(1, min(n_img, n_keep_img))
            final_image_keep = n_keep_img / max(1, n_img)
            num_stages = len(illava_config["illava_llm_k"])
            per_stage_ratio = final_image_keep ** (1.0 / max(1, num_stages))
            illava_config = {
                **illava_config,
                "illava_llm_r": [per_stage_ratio] * num_stages,
                "illava_llm_image_token_start_index": image_start_index,
            }
            kwargs["illava_config"] = illava_config
            kwargs["image_token_length"] = n_img

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Pull through illava_config / image_token_length so the underlying
        # forward is called with these on each step.
        illava_config = kwargs.pop("illava_config", None)
        image_token_length = kwargs.pop("image_token_length", None)
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if illava_config is not None:
            inputs["illava_config"] = illava_config
        if image_token_length is not None:
            inputs["image_token_length"] = image_token_length
        return inputs


AutoConfig.register("llava_llama_training_free", LlavaLlamaTrainingFreeConfig)
AutoModelForCausalLM.register(LlavaLlamaTrainingFreeConfig, LlavaLlamaTrainingFreeForCausalLM)
