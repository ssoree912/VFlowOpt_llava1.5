"""lmms-eval wrapper for LLaVA-1.5 with VFlowOpt-style LLM-side progressive
image-token pruning (training-free).

Mirrors the structure of `llava.py` but routes generate() through the model's
`generate(illava_config=...)` path so that progressive pruning takes effect at
the configured `illava_llm_k` layers. Token importance comes from the LLM's
own self-attention (no precomputed ViT attention map required).

The `keep_ratio` argument is interpreted on the *total token* basis (text +
image), matching the convention used in zap. Per-stage `illava_llm_r` values
are derived dynamically per sample inside the model wrapper.
"""

import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch

torch.backends.cuda.matmul.allow_tf32 = True

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_15_training_free")
class Llava15TrainingFree(lmms):
    """LLaVA-1.5 7B with VFlowOpt-style LLM-side progressive image-token pruning."""

    def __init__(
        self,
        pretrained: str = "/workspace/zap/ckpts/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        # IMPORTANT: model_name must contain both 'llava-v1.5' (so the builder
        # routes to the LLaMA branch) and 'training_free' (so it picks the
        # iLLaVA-enabled subclass). The default works for the standard 7B ckpt.
        model_name: str = "llava-v1.5-7b-training_free",
        attn_implementation: str = best_fit_attn_implementation,
        device_map: str = "cuda:0",
        conv_template: str = "vicuna_v1",
        use_cache: bool = True,
        tie_weights: bool = True,
        truncate_context: bool = False,
        customized_config: Optional[str] = None,
        # iLLaVA / VFlowOpt LLM-side knobs
        keep_ratio: float = 0.5,
        illava_llm_k: str = "10-21",  # comma/dash separated layer indices
        illava_llm_image_token_start_index: int = 35,  # heuristic; overridden per-sample
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {"multimodal": True}
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        try:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained, None, model_name, device_map=self.device_map, **llava_model_args
            )
        except TypeError:
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained, None, model_name, device_map=self.device_map, **llava_model_args
            )

        self._config = self._model.config
        self._model.eval()
        if tie_weights:
            try:
                self._model.tie_weights()
            except Exception:
                pass

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "LLaVA-1.5 training-free wrapper only supports batch_size=1."

        # Parse illava_llm_k.
        if isinstance(illava_llm_k, str):
            parsed_k = [int(x) for x in illava_llm_k.replace(",", "-").split("-") if x.strip()]
        elif isinstance(illava_llm_k, int):
            parsed_k = [illava_llm_k]
        else:
            parsed_k = list(illava_llm_k)
        parsed_k = sorted(parsed_k)

        self.keep_ratio = float(keep_ratio)
        # Pass keep_ratio_total down; per-stage `illava_llm_r` is computed in
        # the model's generate() based on the actual prompt length.
        self.illava_config = {
            "enable_illava_llm": True,
            "enable_illava_vit": False,
            "illava_llm_k": parsed_k,
            "illava_llm_r": [1.0] * len(parsed_k),  # placeholder; recomputed
            "illava_llm_image_token_start_index": illava_llm_image_token_start_index,
            "illava_track_vit_source": False,
            "illava_track_llm_source": False,
            "illava_vit_k": [25],
            "illava_vit_t": 0.0,
            "illava_vit_alpha_v": 0.0,
            "illava_vit_m": 0.0,
            "illava_vit_r": 1.0,
            "keep_ratio_total": self.keep_ratio,
        }

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_kwargs)
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config`"
                    " and set zero stage to 0"
                )
            if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except Exception:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood is not implemented for the iLLaVA wrapper.")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task_name][split_name][ids]) for ids in doc_id]
            flattened_visuals = self.flatten(batched_visuals)
            gen_kwargs = all_gen_kwargs[0]

            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if isinstance(image_tensor, list):
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            question_input = []
            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                question_input.append(conv.get_prompt())

            gen_kwargs["image_sizes"] = [
                flattened_visuals[idx].size for idx in range(len(flattened_visuals))
            ]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")

            input_ids_list = [
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                for prompt in question_input
            ]
            pad_token_ids = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            try:
                with torch.inference_mode():
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        illava_config=copy.deepcopy(self.illava_config),
                    )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating; falling back to empty output for this chunk.")
                text_outputs = [""]

            text_outputs = [t.strip() for t in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("generate_until_multi_round is not implemented for the iLLaVA wrapper.")
