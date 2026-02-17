from typing import Iterator, List, Optional, Tuple, Union, cast

import torch
from compel import Compel, ReturnedEmbeddingsType, SplitLongTextMode
from compel.prompt_parser import Blend, Conjunction, CrossAttentionControlSubstitute, FlattenedPrompt, Fragment
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    OutputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import CLIPField
from invokeai.app.invocations.primitives import ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.ti_utils import generate_ti_list
from invokeai.app.util.prompt_converter import preprocess_prompt_for_invokeai
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    ConditioningFieldData,
    SDXLConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "compel",
    title="Prompt - SD1.5",
    tags=["prompt", "compel"],
    category="conditioning",
    version="1.3.0",
)
class CompelInvocation(BaseInvocation):
    """Parse prompt using compel package to conditioning. Supports A1111 syntax including scheduling."""

    prompt: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )
    clip: CLIPField = InputField(
        title="CLIP",
        description=FieldDescriptions.clip,
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )
    steps: int = InputField(
        default=30,
        ge=1,
        description="Total number of steps (used for prompt scheduling with [from:to:when] syntax)",
    )
    use_a1111_syntax: bool = InputField(
        default=True,
        description="Enable A1111-compatible prompt syntax including (emphasis), [de-emphasis], and [from:to:when] scheduling",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        # Preprocess prompt for A1111 compatibility
        if self.use_a1111_syntax:
            prompt_schedules, has_scheduling = preprocess_prompt_for_invokeai(
                self.prompt, self.steps
            )
        else:
            prompt_schedules = [(self.steps, self.prompt)]
            has_scheduling = False

        def _lora_loader() -> Iterator[Tuple[ModelPatchRaw, float]]:
            for lora in self.clip.loras:
                lora_info = context.models.load(lora.lora)
                assert isinstance(lora_info.model, ModelPatchRaw)
                yield (lora_info.model, lora.weight)
                del lora_info

        text_encoder_info = context.models.load(self.clip.text_encoder)

        # Generate TI list for all prompt variants
        all_prompts = [p[1] for p in prompt_schedules]
        ti_list = generate_ti_list(" ".join(all_prompts), text_encoder_info.config.base, context)

        conditionings: List[BasicConditioningInfo] = []

        with (
            text_encoder_info.model_on_device() as (cached_weights, text_encoder),
            context.models.load(self.clip.tokenizer) as tokenizer,
            LayerPatcher.apply_smart_model_patches(
                model=text_encoder,
                patches=_lora_loader(),
                prefix="lora_te_",
                dtype=text_encoder.dtype,
                cached_weights=cached_weights,
            ),
            ModelPatcher.apply_clip_skip(text_encoder, self.clip.skipped_layers),
            ModelPatcher.apply_ti(tokenizer, text_encoder, ti_list) as (
                patched_tokenizer,
                ti_manager,
            ),
        ):
            context.util.signal_progress("Building conditioning")
            assert isinstance(text_encoder, CLIPTextModel)
            assert isinstance(tokenizer, CLIPTokenizer)

            compel = Compel(
                tokenizer=patched_tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=ti_manager,
                dtype_for_device_getter=TorchDevice.choose_torch_dtype,
                truncate_long_prompts=False,
                device=text_encoder.device,
                split_long_text_mode=SplitLongTextMode.SENTENCES,
            )

            for end_step, prompt_text in prompt_schedules:
                conjunction = Compel.parse_prompt_string(prompt_text)

                if context.config.get().log_tokenization:
                    log_tokenization_for_conjunction(conjunction, patched_tokenizer)

                c, _options = compel.build_conditioning_tensor_for_conjunction(conjunction)
                c = c.detach().to("cpu")

                conditionings.append(BasicConditioningInfo(embeds=c, end_at_step=end_step))

        del compel
        del patched_tokenizer
        del tokenizer
        del ti_manager
        del text_encoder
        del text_encoder_info

        conditioning_data = ConditioningFieldData(conditionings=conditionings)
        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
                mask=self.mask,
            )
        )


class SDXLPromptInvocationBase:
    """Prompt processor for SDXL models with A1111 syntax support."""

    def run_clip_compel(
        self,
        context: InvocationContext,
        clip_field: CLIPField,
        prompt: str,
        get_pooled: bool,
        lora_prefix: str,
        zero_on_empty: bool,
        steps: int = 30,
        use_a1111_syntax: bool = True,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Returns:
            Tuple of:
            - List of (end_step, conditioning_tensor) for scheduled prompts
            - Pooled embeddings (if requested)
        """

        # Preprocess for A1111 syntax
        if use_a1111_syntax:
            prompt_schedules, _ = preprocess_prompt_for_invokeai(prompt, steps)
        else:
            prompt_schedules = [(steps, prompt)]

        text_encoder_info = context.models.load(clip_field.text_encoder)

        # Return zero on empty
        if prompt == "" and zero_on_empty:
            cpu_text_encoder = text_encoder_info.model
            assert isinstance(cpu_text_encoder, torch.nn.Module)
            c = torch.zeros(
                (
                    1,
                    cpu_text_encoder.config.max_position_embeddings,
                    cpu_text_encoder.config.hidden_size,
                ),
                dtype=cpu_text_encoder.dtype,
            )
            if get_pooled:
                c_pooled = torch.zeros(
                    (1, cpu_text_encoder.config.hidden_size),
                    dtype=c.dtype,
                )
            else:
                c_pooled = None
            return [(steps, c)], c_pooled

        def _lora_loader() -> Iterator[Tuple[ModelPatchRaw, float]]:
            for lora in clip_field.loras:
                lora_info = context.models.load(lora.lora)
                lora_model = lora_info.model
                assert isinstance(lora_model, ModelPatchRaw)
                yield (lora_model, lora.weight)
                del lora_info

        all_prompts = [p[1] for p in prompt_schedules]
        ti_list = generate_ti_list(" ".join(all_prompts), text_encoder_info.config.base, context)

        scheduled_conds: List[Tuple[int, torch.Tensor]] = []
        c_pooled: Optional[torch.Tensor] = None

        with (
            text_encoder_info.model_on_device() as (cached_weights, text_encoder),
            context.models.load(clip_field.tokenizer) as tokenizer,
            LayerPatcher.apply_smart_model_patches(
                model=text_encoder,
                patches=_lora_loader(),
                prefix=lora_prefix,
                dtype=text_encoder.dtype,
                cached_weights=cached_weights,
            ),
            ModelPatcher.apply_clip_skip(text_encoder, clip_field.skipped_layers),
            ModelPatcher.apply_ti(tokenizer, text_encoder, ti_list) as (
                patched_tokenizer,
                ti_manager,
            ),
        ):
            context.util.signal_progress("Building conditioning")
            assert isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection))
            assert isinstance(tokenizer, CLIPTokenizer)

            text_encoder = cast(CLIPTextModel, text_encoder)
            compel = Compel(
                tokenizer=patched_tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=ti_manager,
                dtype_for_device_getter=TorchDevice.choose_torch_dtype,
                truncate_long_prompts=False,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=get_pooled,
                device=text_encoder.device,
                split_long_text_mode=SplitLongTextMode.SENTENCES,
            )

            for end_step, prompt_text in prompt_schedules:
                conjunction = Compel.parse_prompt_string(prompt_text)

                if context.config.get().log_tokenization:
                    log_tokenization_for_conjunction(conjunction, patched_tokenizer)

                c, _options = compel.build_conditioning_tensor_for_conjunction(conjunction)
                c = c.detach().to("cpu")
                scheduled_conds.append((end_step, c))

            # Get pooled embeddings from the last prompt
            if get_pooled:
                c_pooled = compel.conditioning_provider.get_pooled_embeddings([prompt_schedules[-1][1]])
                c_pooled = c_pooled.detach().to("cpu")

        del compel
        del patched_tokenizer
        del tokenizer
        del ti_manager
        del text_encoder
        del text_encoder_info

        return scheduled_conds, c_pooled


@invocation(
    "sdxl_compel_prompt",
    title="Prompt - SDXL",
    tags=["sdxl", "compel", "prompt"],
    category="conditioning",
    version="1.3.0",
)
class SDXLCompelPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Parse prompt using compel package to conditioning. Supports A1111 syntax including scheduling."""

    prompt: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )
    style: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )
    original_width: int = InputField(default=1024, description="")
    original_height: int = InputField(default=1024, description="")
    crop_top: int = InputField(default=0, description="")
    crop_left: int = InputField(default=0, description="")
    target_width: int = InputField(default=1024, description="")
    target_height: int = InputField(default=1024, description="")
    clip: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection, title="CLIP 1")
    clip2: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection, title="CLIP 2")
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )
    steps: int = InputField(
        default=30,
        ge=1,
        description="Total number of steps (used for prompt scheduling)",
    )
    use_a1111_syntax: bool = InputField(
        default=True,
        description="Enable A1111-compatible prompt syntax",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        c1_scheduled, c1_pooled = self.run_clip_compel(
            context, self.clip, self.prompt, False, "lora_te1_",
            zero_on_empty=True, steps=self.steps, use_a1111_syntax=self.use_a1111_syntax
        )

        style_prompt = self.style.strip() if self.style.strip() else self.prompt
        c2_scheduled, c2_pooled = self.run_clip_compel(
            context, self.clip2, style_prompt, True, "lora_te2_",
            zero_on_empty=True, steps=self.steps, use_a1111_syntax=self.use_a1111_syntax
        )

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)
        target_size = (self.target_height, self.target_width)
        add_time_ids = torch.tensor([original_size + crop_coords + target_size])

        # Build scheduled conditionings - merge schedules from c1 and c2
        conditionings: List[SDXLConditioningInfo] = []
        all_steps = sorted(set([s[0] for s in c1_scheduled] + [s[0] for s in c2_scheduled]))

        for target_step in all_steps:
            # Find appropriate c1 for this step
            c1 = None
            for end_step, cond in c1_scheduled:
                if target_step <= end_step:
                    c1 = cond
                    break
            if c1 is None:
                c1 = c1_scheduled[-1][1]

            # Find appropriate c2 for this step
            c2 = None
            for end_step, cond in c2_scheduled:
                if target_step <= end_step:
                    c2 = cond
                    break
            if c2 is None:
                c2 = c2_scheduled[-1][1]

            # Pad to match dimensions
            if c1.shape[1] < c2.shape[1]:
                c1 = torch.cat([
                    c1,
                    torch.zeros((c1.shape[0], c2.shape[1] - c1.shape[1], c1.shape[2]), device=c1.device, dtype=c1.dtype),
                ], dim=1)
            elif c1.shape[1] > c2.shape[1]:
                c2 = torch.cat([
                    c2,
                    torch.zeros((c2.shape[0], c1.shape[1] - c2.shape[1], c2.shape[2]), device=c2.device, dtype=c2.dtype),
                ], dim=1)

            assert c2_pooled is not None
            conditionings.append(
                SDXLConditioningInfo(
                    embeds=torch.cat([c1, c2], dim=-1),
                    pooled_embeds=c2_pooled,
                    add_time_ids=add_time_ids,
                    end_at_step=target_step,
                )
            )

        conditioning_data = ConditioningFieldData(conditionings=conditionings)
        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
                mask=self.mask,
            )
        )


@invocation(
    "sdxl_refiner_compel_prompt",
    title="Prompt - SDXL Refiner",
    tags=["sdxl", "compel", "prompt"],
    category="conditioning",
    version="1.2.0",
)
class SDXLRefinerCompelPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    """Parse prompt using compel package to conditioning. Supports A1111 syntax."""

    style: str = InputField(
        default="",
        description=FieldDescriptions.compel_prompt,
        ui_component=UIComponent.Textarea,
    )
    original_width: int = InputField(default=1024, description="")
    original_height: int = InputField(default=1024, description="")
    crop_top: int = InputField(default=0, description="")
    crop_left: int = InputField(default=0, description="")
    aesthetic_score: float = InputField(default=6.0, description=FieldDescriptions.sdxl_aesthetic)
    clip2: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection)
    steps: int = InputField(
        default=30,
        ge=1,
        description="Total number of steps (used for prompt scheduling)",
    )
    use_a1111_syntax: bool = InputField(
        default=True,
        description="Enable A1111-compatible prompt syntax",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        c2_scheduled, c2_pooled = self.run_clip_compel(
            context, self.clip2, self.style, True, "<NONE>",
            zero_on_empty=False, steps=self.steps, use_a1111_syntax=self.use_a1111_syntax
        )

        original_size = (self.original_height, self.original_width)
        crop_coords = (self.crop_top, self.crop_left)
        add_time_ids = torch.tensor([original_size + crop_coords + (self.aesthetic_score,)])

        conditionings: List[SDXLConditioningInfo] = []
        for end_step, c2 in c2_scheduled:
            assert c2_pooled is not None
            conditionings.append(
                SDXLConditioningInfo(
                    embeds=c2,
                    pooled_embeds=c2_pooled,
                    add_time_ids=add_time_ids,
                    end_at_step=end_step,
                )
            )

        conditioning_data = ConditioningFieldData(conditionings=conditionings)
        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput.build(conditioning_name)


@invocation_output("clip_skip_output")
class CLIPSkipInvocationOutput(BaseInvocationOutput):
    """CLIP skip node output"""
    clip: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP")

@invocation(
    "clip_skip",
    title="Apply CLIP Skip - SD1.5, SDXL",
    tags=["clipskip", "clip", "skip"],
    category="conditioning",
    version="1.1.1",
)
class CLIPSkipInvocation(BaseInvocation):
    """Skip layers in clip text_encoder model."""

    clip: CLIPField = InputField(description=FieldDescriptions.clip, input=Input.Connection, title="CLIP")
    skipped_layers: int = InputField(default=0, ge=0, description=FieldDescriptions.skipped_layers)

    def invoke(self, context: InvocationContext) -> CLIPSkipInvocationOutput:
        self.clip.skipped_layers += self.skipped_layers
        return CLIPSkipInvocationOutput(clip=self.clip)


# Utility functions

def get_max_token_count(
    tokenizer: CLIPTokenizer,
    prompt: Union[FlattenedPrompt, Blend, Conjunction],
    truncate_if_too_long: bool = False,
) -> int:
    if type(prompt) is Blend:
        blend: Blend = prompt
        return max([get_max_token_count(tokenizer, p, truncate_if_too_long) for p in blend.prompts])
    elif type(prompt) is Conjunction:
        conjunction: Conjunction = prompt
        return sum([get_max_token_count(tokenizer, p, truncate_if_too_long) for p in conjunction.prompts])
    else:
        return len(get_tokens_for_prompt_object(tokenizer, prompt, truncate_if_too_long))


def get_tokens_for_prompt_object(
    tokenizer: CLIPTokenizer, parsed_prompt: FlattenedPrompt, truncate_if_too_long: bool = True
) -> List[str]:
    if type(parsed_prompt) is Blend:
        raise ValueError("Blend is not supported here - you need to get tokens for each of its .children")

    text_fragments = [
        (
            x.text
            if type(x) is Fragment
            else (" ".join([f.text for f in x.original]) if type(x) is CrossAttentionControlSubstitute else str(x))
        )
        for x in parsed_prompt.children
    ]
    text = " ".join(text_fragments)
    tokens: List[str] = tokenizer.tokenize(text)
    if truncate_if_too_long:
        max_tokens_length = tokenizer.model_max_length - 2
        tokens = tokens[0:max_tokens_length]
    return tokens


def log_tokenization_for_conjunction(
    c: Conjunction, tokenizer: CLIPTokenizer, display_label_prefix: Optional[str] = None
) -> None:
    display_label_prefix = display_label_prefix or ""
    for i, p in enumerate(c.prompts):
        if len(c.prompts) > 1:
            this_display_label_prefix = f"{display_label_prefix}(conjunction part {i + 1}, weight={c.weights[i]})"
        else:
            this_display_label_prefix = display_label_prefix
        log_tokenization_for_prompt_object(p, tokenizer, display_label_prefix=this_display_label_prefix)


def log_tokenization_for_prompt_object(
    p: Union[Blend, FlattenedPrompt], tokenizer: CLIPTokenizer, display_label_prefix: Optional[str] = None
) -> None:
    display_label_prefix = display_label_prefix or ""
    if type(p) is Blend:
        blend: Blend = p
        for i, c in enumerate(blend.prompts):
            log_tokenization_for_prompt_object(
                c, tokenizer,
                display_label_prefix=f"{display_label_prefix}(blend part {i + 1}, weight={blend.weights[i]})",
            )
    elif type(p) is FlattenedPrompt:
        flattened_prompt: FlattenedPrompt = p
        if flattened_prompt.wants_cross_attention_control:
            original_fragments = []
            edited_fragments = []
            for f in flattened_prompt.children:
                if type(f) is CrossAttentionControlSubstitute:
                    original_fragments += f.original
                    edited_fragments += f.edited
                else:
                    original_fragments.append(f)
                    edited_fragments.append(f)

            original_text = " ".join([x.text for x in original_fragments])
            log_tokenization_for_text(original_text, tokenizer, display_label=f"{display_label_prefix}(.swap originals)")
            edited_text = " ".join([x.text for x in edited_fragments])
            log_tokenization_for_text(edited_text, tokenizer, display_label=f"{display_label_prefix}(.swap replacements)")
        else:
            text = " ".join([x.text for x in flattened_prompt.children])
            log_tokenization_for_text(text, tokenizer, display_label=display_label_prefix)


def log_tokenization_for_text(
    text: str,
    tokenizer: CLIPTokenizer,
    display_label: Optional[str] = None,
    truncate_if_too_long: Optional[bool] = False,
) -> None:
    tokens = tokenizer.tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)

    for i in range(0, totalTokens):
        token = tokens[i].replace("</w>", " ")
        s = (usedTokens % 6) + 1
        if truncate_if_too_long and i >= tokenizer.model_max_length:
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
        else:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1

    if usedTokens > 0:
        print(f"\n>> [TOKENLOG] Tokens {display_label or ''} ({usedTokens}):")
        print(f"{tokenized}\x1b[0m")

    if discarded != "":
        print(f"\n>> [TOKENLOG] Tokens Discarded ({totalTokens - usedTokens}):")
        print(f"{discarded}\x1b[0m")
