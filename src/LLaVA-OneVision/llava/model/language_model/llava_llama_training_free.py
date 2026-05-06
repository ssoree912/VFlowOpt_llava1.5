import json
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

from .modeling_llama import LlamaForCausalLM, LlamaModel


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
        self.default_illava_config = None
        self._active_illava_config = None
        self._last_image_token_length = 0
        self.post_init()

    def get_model(self):
        return self.model

    def _get_image_token_start(self, input_ids, attention_mask):
        if input_ids is None or input_ids.shape[0] == 0:
            return 0

        cur_input_ids = input_ids[0]
        if attention_mask is not None:
            cur_input_ids = cur_input_ids[attention_mask[0].bool()]

        image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        if image_positions.numel() == 0:
            return 0
        return int(image_positions[0].item())

    def _make_vit_attention_map(self, image_forward_outs, image_token_length, illava_config, device):
        if image_token_length <= 0:
            return torch.ones(1, device=device, dtype=torch.float32)

        attentions = getattr(image_forward_outs, "attentions", None)
        if not attentions:
            return torch.ones(image_token_length, device=device, dtype=torch.float32)

        vit_k = illava_config.get("illava_vit_k", [23]) if illava_config else [23]
        if isinstance(vit_k, (list, tuple)):
            layer_idx = int(vit_k[-1])
        else:
            layer_idx = int(vit_k)
        layer_idx = max(0, min(layer_idx, len(attentions) - 1))

        attn = attentions[layer_idx]
        if attn is None:
            return torch.ones(image_token_length, device=device, dtype=torch.float32)

        attn = attn[0].float().mean(dim=0)
        received = attn.mean(dim=0)
        threshold = float(illava_config.get("illava_vit_t", 5.0)) if illava_config else 5.0
        global_indices = torch.nonzero(received > threshold * received.mean(), as_tuple=False).flatten()
        if global_indices.numel() == 0:
            combined = received
        else:
            combined = attn[global_indices].mean(dim=0)

        if combined.numel() == image_token_length + 1:
            combined = combined[1:]
        elif combined.numel() > image_token_length:
            combined = combined[-image_token_length:]
        elif combined.numel() < image_token_length:
            pad = torch.ones(image_token_length - combined.numel(), device=combined.device, dtype=combined.dtype)
            combined = torch.cat([combined, pad], dim=0)

        return combined.to(device=device, dtype=torch.float32)

    def encode_images(self, images):
        illava_config = self._active_illava_config
        needs_attn = illava_config is not None and (
            illava_config.get("enable_illava_llm", False) or illava_config.get("enable_illava_vit", False)
        )

        vision_tower = self.get_model().get_vision_tower()
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=vision_tower.device, dtype=vision_tower.dtype),
            output_hidden_states=True,
            output_attentions=needs_attn,
        )
        image_features = vision_tower.feature_select(image_forward_outs).to(images.dtype)
        self._last_image_token_length = int(image_features.shape[1])

        if needs_attn:
            attn_map = self._make_vit_attention_map(image_forward_outs, self._last_image_token_length, illava_config, image_features.device)
            os.environ["VIT_ATTN_MAP"] = json.dumps(attn_map.detach().cpu().tolist())

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
        illava_config=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, 0

        image_token_start = self._get_image_token_start(input_ids, attention_mask)
        self._last_image_token_length = 0
        self._active_illava_config = illava_config
        try:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
            )
        finally:
            self._active_illava_config = None

        image_token_length = self._last_image_token_length
        if illava_config is not None:
            illava_config["illava_llm_image_token_start_index"] = image_token_start
            illava_config["image_token_length"] = image_token_length

        return input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, None, image_token_length

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
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        illava_config=None,
        source_indice_list=None,
        raw_frames=None,
        image_token_length=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if illava_config is None:
            illava_config = self.default_illava_config

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                source_indice_list,
                prepared_image_token_length,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes=image_sizes,
                illava_config=illava_config,
            )
            image_token_length = prepared_image_token_length

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

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
                cache_position=cache_position,
                illava_config=illava_config,
                source_indice_list=source_indice_list,
                raw_frames=raw_frames,
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
            cache_position=cache_position,
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
        if illava_config is None:
            illava_config = self.default_illava_config

        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        source_indice_list = None
        image_token_length = 0
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                source_indice_list,
                image_token_length,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                illava_config=illava_config,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            illava_config=illava_config,
            source_indice_list=source_indice_list,
            image_token_length=image_token_length,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        illava_config = kwargs.pop("illava_config", None)
        source_indice_list = kwargs.pop("source_indice_list", None)
        raw_frames = kwargs.pop("raw_frames", None)
        image_token_length = kwargs.pop("image_token_length", None)
        effective_past_key_values = past_key_values
        if past_key_values is not None and inputs_embeds is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]
            if past_length == 0:
                effective_past_key_values = None
        debug_shapes = os.environ.get("ILLAVA_DEBUG_SHAPES", "0") == "1"
        if debug_shapes:
            embed_shape = None if inputs_embeds is None else tuple(inputs_embeds.shape)
            print(
                f"[illava-llama] prepare_gen input_ids={tuple(input_ids.shape)} "
                f"inputs_embeds={embed_shape} past={past_key_values is not None} "
                f"image_token_length={image_token_length}",
                flush=True,
            )

        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=effective_past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if effective_past_key_values is not None and "input_ids" in inputs and inputs["input_ids"].shape[1] == 0:
            if hasattr(effective_past_key_values, "get_seq_length"):
                past_length = effective_past_key_values.get_seq_length()
            else:
                past_length = effective_past_key_values[0][0].shape[2]
            inputs["input_ids"] = input_ids[:, -1:].contiguous()
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                inputs["position_ids"] = position_ids[:, -1:].contiguous()
            else:
                inputs["position_ids"] = torch.full((input_ids.shape[0], 1), past_length, dtype=torch.long, device=input_ids.device)
            inputs["cache_position"] = torch.arange(past_length, past_length + 1, device=input_ids.device)

        query_length = inputs["inputs_embeds"].shape[1] if "inputs_embeds" in inputs else inputs["input_ids"].shape[1]
        if inputs.get("position_ids", None) is not None and inputs["position_ids"].shape[-1] != query_length:
            inputs["position_ids"] = inputs["position_ids"][:, -query_length:].contiguous()
        if inputs.get("cache_position", None) is not None and inputs["cache_position"].numel() != query_length:
            if effective_past_key_values is not None:
                if hasattr(effective_past_key_values, "get_seq_length"):
                    past_length = effective_past_key_values.get_seq_length()
                else:
                    past_length = effective_past_key_values[0][0].shape[2]
            else:
                past_length = 0
            inputs["cache_position"] = torch.arange(past_length, past_length + query_length, device=input_ids.device)

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if illava_config is not None:
            inputs["illava_config"] = illava_config
        if source_indice_list is not None:
            inputs["source_indice_list"] = source_indice_list
        if raw_frames is not None:
            inputs["raw_frames"] = raw_frames
        if image_token_length is not None:
            inputs["image_token_length"] = image_token_length
        if debug_shapes:
            out_embed_shape = None if inputs.get("inputs_embeds", None) is None else tuple(inputs["inputs_embeds"].shape)
            out_ids_shape = None if inputs.get("input_ids", None) is None else tuple(inputs["input_ids"].shape)
            out_pos_shape = None if inputs.get("position_ids", None) is None else tuple(inputs["position_ids"].shape)
            print(
                f"[illava-llama] prepare_gen_out input_ids={out_ids_shape} "
                f"inputs_embeds={out_embed_shape} position_ids={out_pos_shape} "
                f"cache={tuple(inputs['cache_position'].shape) if inputs.get('cache_position', None) is not None else None}",
                flush=True,
            )
        return inputs


AutoConfig.register("llava_llama_training_free", LlavaLlamaTrainingFreeConfig)
AutoModelForCausalLM.register(LlavaLlamaTrainingFreeConfig, LlavaLlamaTrainingFreeForCausalLM)
