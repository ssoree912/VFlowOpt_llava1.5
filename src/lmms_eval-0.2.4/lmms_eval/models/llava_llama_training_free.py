import math
import re
from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.llava import Llava, best_fit_attn_implementation


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_layers(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(layer) for layer in value]
    parts = [part for part in re.split(r"[-,]", str(value)) if part != ""]
    return [int(part) for part in parts]


def _parse_ratios(value):
    if value is None or str(value).strip().lower() in {"", "none", "auto"}:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(ratio) for ratio in value]
    parts = [part for part in re.split(r"[-,]", str(value)) if part != ""]
    return [float(part) for part in parts]


@register_model("llava_llama_training_free")
class LlavaLlamaTrainingFree(Llava):
    def __init__(
        self,
        pretrained: str = "/workspace/look-m/models/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = "llava_llama_training_free",
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "vicuna_v1",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,
        customized_config=None,
        enable_illava_vit: Optional[bool] = True,
        illava_vit_k: Optional[str] = "23",
        illava_vit_t: Optional[float] = 5.0,
        illava_vit_alpha_v: Optional[float] = 0.0,
        enable_illava_llm: Optional[bool] = True,
        illava_llm_k: Optional[str] = "9-18",
        illava_llm_r=None,
        illava_keep_ratio: Optional[float] = 0.30,
        illava_total_keep_ratio: Optional[bool] = True,
        illava_track_vit_source: Optional[bool] = False,
        illava_track_llm_source: Optional[bool] = False,
        **kwargs,
    ) -> None:
        self.illava_keep_ratio = float(illava_keep_ratio)
        enable_illava_vit = _as_bool(enable_illava_vit)
        enable_illava_llm = _as_bool(enable_illava_llm)
        illava_total_keep_ratio = _as_bool(illava_total_keep_ratio)
        illava_llm_k = _parse_layers(illava_llm_k) or [18]
        illava_vit_k = _parse_layers(illava_vit_k) or [23]

        parsed_llm_r = _parse_ratios(illava_llm_r)
        if parsed_llm_r is None:
            if illava_total_keep_ratio:
                stage_keep_ratio = self.illava_keep_ratio ** (1.0 / max(1, len(illava_llm_k)))
                parsed_llm_r = [stage_keep_ratio for _ in illava_llm_k]
            else:
                parsed_llm_r = [self.illava_keep_ratio for _ in illava_llm_k]
        elif len(parsed_llm_r) == 1 and len(illava_llm_k) > 1:
            parsed_llm_r = [parsed_llm_r[0] for _ in illava_llm_k]

        if len(parsed_llm_r) != len(illava_llm_k):
            raise ValueError(f"illava_llm_r length ({len(parsed_llm_r)}) must match illava_llm_k length ({len(illava_llm_k)})")

        self.illava_config = {
            "illava_keep_ratio": self.illava_keep_ratio,
            "illava_total_keep_ratio": illava_total_keep_ratio,
            "enable_illava_vit": enable_illava_vit,
            "illava_vit_k": illava_vit_k,
            "illava_vit_t": float(illava_vit_t),
            "illava_vit_alpha_v": float(illava_vit_alpha_v),
            "enable_illava_llm": enable_illava_llm,
            "illava_llm_k": illava_llm_k,
            "illava_llm_r": parsed_llm_r,
            "illava_llm_image_token_start_index": 0,
            "illava_track_vit_source": _as_bool(illava_track_vit_source),
            "illava_track_llm_source": _as_bool(illava_track_llm_source),
        }

        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            customized_config=customized_config,
            **kwargs,
        )

        self.model.default_illava_config = self.illava_config
