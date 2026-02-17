from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData

if TYPE_CHECKING:
    from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
    from invokeai.backend.stable_diffusion.denoise_context import UNetKwargs


@dataclass
class BasicConditioningInfo:
    """SD 1/2 text conditioning information produced by Compel."""

    embeds: torch.Tensor
    end_at_step: Optional[int] = None  # For A1111-style prompt scheduling

    def to(self, device, dtype=None):
        self.embeds = self.embeds.to(device=device, dtype=dtype)
        return self


@dataclass
class SDXLConditioningInfo(BasicConditioningInfo):
    """SDXL text conditioning information produced by Compel."""

    pooled_embeds: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    add_time_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # end_at_step inherited from BasicConditioningInfo

    def to(self, device, dtype=None):
        self.pooled_embeds = self.pooled_embeds.to(device=device, dtype=dtype)
        self.add_time_ids = self.add_time_ids.to(device=device, dtype=dtype)
        return super().to(device=device, dtype=dtype)


@dataclass
class FLUXConditioningInfo:
    clip_embeds: torch.Tensor
    t5_embeds: torch.Tensor
    end_at_step: Optional[int] = None  # For A1111-style prompt scheduling

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.clip_embeds = self.clip_embeds.to(device=device, dtype=dtype)
        self.t5_embeds = self.t5_embeds.to(device=device, dtype=dtype)
        return self


@dataclass
class SD3ConditioningInfo:
    clip_l_pooled_embeds: torch.Tensor
    clip_l_embeds: torch.Tensor
    clip_g_pooled_embeds: torch.Tensor
    clip_g_embeds: torch.Tensor
    t5_embeds: torch.Tensor | None
    end_at_step: Optional[int] = None  # For A1111-style prompt scheduling

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.clip_l_pooled_embeds = self.clip_l_pooled_embeds.to(device=device, dtype=dtype)
        self.clip_l_embeds = self.clip_l_embeds.to(device=device, dtype=dtype)
        self.clip_g_pooled_embeds = self.clip_g_pooled_embeds.to(device=device, dtype=dtype)
        self.clip_g_embeds = self.clip_g_embeds.to(device=device, dtype=dtype)
        if self.t5_embeds is not None:
            self.t5_embeds = self.t5_embeds.to(device=device, dtype=dtype)
        return self


@dataclass
class CogView4ConditioningInfo:
    glm_embeds: torch.Tensor
    end_at_step: Optional[int] = None  # For A1111-style prompt scheduling

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.glm_embeds = self.glm_embeds.to(device=device, dtype=dtype)
        return self


@dataclass
class ZImageConditioningInfo:
    """Z-Image text conditioning information from Qwen3 text encoder."""

    prompt_embeds: torch.Tensor
    """Text embeddings from Qwen3 encoder. Shape: (batch_size, seq_len, hidden_size)."""
    end_at_step: Optional[int] = None  # For A1111-style prompt scheduling

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.prompt_embeds = self.prompt_embeds.to(device=device, dtype=dtype)
        return self


# Type alias for all conditioning info types
ConditioningInfo = Union[
    BasicConditioningInfo,
    SDXLConditioningInfo,
    FLUXConditioningInfo,
    SD3ConditioningInfo,
    CogView4ConditioningInfo,
    ZImageConditioningInfo,
]


@dataclass
class ConditioningFieldData:
    # If you change this class, adding more types, you _must_ update the instantiation of ObjectSerializerDisk in
    # invokeai/app/api/dependencies.py, adding the types to the list of safe globals. If you do not, torch will be
    # unable to deserialize the object and will raise an error.
    conditionings: (
        List[BasicConditioningInfo]
        | List[SDXLConditioningInfo]
        | List[FLUXConditioningInfo]
        | List[SD3ConditioningInfo]
        | List[CogView4ConditioningInfo]
        | List[ZImageConditioningInfo]
    )

    def get_conditioning_for_step(self, step: int) -> ConditioningInfo:
        """
        Get the appropriate conditioning for a given denoising step.
        Supports A1111-style prompt scheduling where different prompts
        can be active at different steps.

        Args:
            step: Current step (1-indexed)

        Returns:
            The conditioning info to use for this step
        """
        if len(self.conditionings) == 1:
            return self.conditionings[0]

        for cond in self.conditionings:
            if cond.end_at_step is not None and step <= cond.end_at_step:
                return cond

        # Return the last one if no match
        return self.conditionings[-1]

    @property
    def has_scheduling(self) -> bool:
        """Returns True if this conditioning has multiple scheduled prompts."""
        return len(self.conditionings) > 1


@dataclass
class IPAdapterConditioningInfo:
    cond_image_prompt_embeds: torch.Tensor
    """IP-Adapter image encoder conditioning embeddings.
    Shape: (num_images, num_tokens, encoding_dim).
    """
    uncond_image_prompt_embeds: torch.Tensor
    """IP-Adapter image encoding embeddings to use for unconditional generation.
    Shape: (num_images, num_tokens, encoding_dim).
    """


@dataclass
class IPAdapterData:
    """Data class for IP-Adapter configuration.

    Attributes:
        ip_adapter_model: The IP-Adapter model to use.
        ip_adapter_conditioning: The IP-Adapter conditioning data.
        mask: The mask to apply to the IP-Adapter conditioning.
        target_blocks: List of target attention block names to apply IP-Adapter to.
        negative_blocks: List of target attention block names that should use negative attention.
        weight: The weight to apply to the IP-Adapter conditioning.
        begin_step_percent: The percentage of steps at which to start applying the IP-Adapter.
        end_step_percent: The percentage of steps at which to stop applying the IP-Adapter.
        method: The method to use for applying the IP-Adapter ('full', 'style', 'composition').
    """

    ip_adapter_model: IPAdapter
    ip_adapter_conditioning: IPAdapterConditioningInfo
    mask: torch.Tensor
    target_blocks: List[str]
    negative_blocks: List[str] = field(default_factory=list)
    weight: Union[float, List[float]] = 1.0
    begin_step_percent: float = 0.0
    end_step_percent: float = 1.0
    method: str = "full"

    def scale_for_step(self, step_index: int, total_steps: int) -> float:
        first_adapter_step = math.floor(self.begin_step_percent * total_steps)
        last_adapter_step = math.ceil(self.end_step_percent * total_steps)
        weight = self.weight[step_index] if isinstance(self.weight, List) else self.weight
        if step_index >= first_adapter_step and step_index <= last_adapter_step:
            # Only apply this IP-Adapter if the current step is within the IP-Adapter's begin/end step range.
            return weight
        # Otherwise, set the IP-Adapter's scale to 0, so it has no effect.
        return 0.0


@dataclass
class Range:
    start: int
    end: int


class TextConditioningRegions:
    def __init__(
        self,
        masks: torch.Tensor,
        ranges: list[Range],
    ):
        # A binary mask indicating the regions of the image that the prompt should be applied to.
        # Shape: (1, num_prompts, height, width)
        # Dtype: torch.bool
        self.masks = masks

        # A list of ranges indicating the start and end indices of the embeddings that corresponding mask applies to.
        # ranges[i] contains the embedding range for the i'th prompt / mask.
        self.ranges = ranges

        assert self.masks.shape[1] == len(self.ranges)


class ConditioningMode(Enum):
    Both = "both"
    Negative = "negative"
    Positive = "positive"


# Add to the existing TextConditioningData class in conditioning_data.py

class TextConditioningData:
    def __init__(
        self,
        uncond_text: Union[BasicConditioningInfo, SDXLConditioningInfo],
        cond_text: Union[BasicConditioningInfo, SDXLConditioningInfo],
        uncond_regions: Optional[TextConditioningRegions],
        cond_regions: Optional[TextConditioningRegions],
        guidance_scale: Union[float, List[float]],
        guidance_rescale_multiplier: float = 0,
    ):
        self.uncond_text = uncond_text
        self.cond_text = cond_text
        self.uncond_regions = uncond_regions
        self.cond_regions = cond_regions
        self.guidance_scale = guidance_scale
        self.guidance_rescale_multiplier = guidance_rescale_multiplier

    def is_sdxl(self):
        assert isinstance(self.uncond_text, SDXLConditioningInfo) == isinstance(self.cond_text, SDXLConditioningInfo)
        return isinstance(self.cond_text, SDXLConditioningInfo)

    @staticmethod
    def from_field_data(
        uncond_field_data: ConditioningFieldData,
        cond_field_data: ConditioningFieldData,
        uncond_regions: Optional[TextConditioningRegions],
        cond_regions: Optional[TextConditioningRegions],
        guidance_scale: Union[float, List[float]],
        guidance_rescale_multiplier: float = 0,
        current_step: Optional[int] = None,
    ) -> Union["TextConditioningData", "ScheduledTextConditioningData"]:
        """
        Create a TextConditioningData from ConditioningFieldData objects.

        If the field data contains scheduled prompts, returns a ScheduledTextConditioningData.
        Otherwise returns a regular TextConditioningData.

        Args:
            uncond_field_data: Unconditional conditioning field data
            cond_field_data: Conditional conditioning field data
            uncond_regions: Unconditional regional prompting data
            cond_regions: Conditional regional prompting data
            guidance_scale: CFG guidance scale
            guidance_rescale_multiplier: Rescale multiplier for CFG
            current_step: If provided and scheduling is present, returns conditioning for this specific step

        Returns:
            TextConditioningData or ScheduledTextConditioningData
        """
        # Check if either conditioning has scheduling
        has_scheduling = uncond_field_data.has_scheduling or cond_field_data.has_scheduling

        if not has_scheduling:
            # No scheduling, return simple TextConditioningData
            return TextConditioningData(
                uncond_text=uncond_field_data.conditionings[0],
                cond_text=cond_field_data.conditionings[0],
                uncond_regions=uncond_regions,
                cond_regions=cond_regions,
                guidance_scale=guidance_scale,
                guidance_rescale_multiplier=guidance_rescale_multiplier,
            )

        if current_step is not None:
            # Scheduling is present but we want a specific step
            return TextConditioningData(
                uncond_text=uncond_field_data.get_conditioning_for_step(current_step),
                cond_text=cond_field_data.get_conditioning_for_step(current_step),
                uncond_regions=uncond_regions,
                cond_regions=cond_regions,
                guidance_scale=guidance_scale,
                guidance_rescale_multiplier=guidance_rescale_multiplier,
            )

        # Scheduling is present and no specific step requested, return scheduled version
        return ScheduledTextConditioningData(
            uncond_field_data=uncond_field_data,
            cond_field_data=cond_field_data,
            uncond_regions=uncond_regions,
            cond_regions=cond_regions,
            guidance_scale=guidance_scale,
            guidance_rescale_multiplier=guidance_rescale_multiplier,
        )

    # ... rest of existing methods ...

    def to_unet_kwargs(self, unet_kwargs: UNetKwargs, conditioning_mode: ConditioningMode):
        """Fills unet arguments with data from provided conditionings.

        Args:
            unet_kwargs (UNetKwargs): Object which stores UNet model arguments.
            conditioning_mode (ConditioningMode): Describes which conditionings should be used.
        """
        _, _, h, w = unet_kwargs.sample.shape
        device = unet_kwargs.sample.device
        dtype = unet_kwargs.sample.dtype

        # TODO: combine regions with conditionings
        if conditioning_mode == ConditioningMode.Both:
            conditionings = [self.uncond_text, self.cond_text]
            c_regions = [self.uncond_regions, self.cond_regions]
        elif conditioning_mode == ConditioningMode.Positive:
            conditionings = [self.cond_text]
            c_regions = [self.cond_regions]
        elif conditioning_mode == ConditioningMode.Negative:
            conditionings = [self.uncond_text]
            c_regions = [self.uncond_regions]
        else:
            raise ValueError(f"Unexpected conditioning mode: {conditioning_mode}")

        encoder_hidden_states, encoder_attention_mask = self._concat_conditionings_for_batch(
            [c.embeds for c in conditionings]
        )

        unet_kwargs.encoder_hidden_states = encoder_hidden_states
        unet_kwargs.encoder_attention_mask = encoder_attention_mask

        if self.is_sdxl():
            added_cond_kwargs = dict(  # noqa: C408
                text_embeds=torch.cat([c.pooled_embeds for c in conditionings]),
                time_ids=torch.cat([c.add_time_ids for c in conditionings]),
            )

            unet_kwargs.added_cond_kwargs = added_cond_kwargs

        if any(r is not None for r in c_regions):
            tmp_regions = []
            for c, r in zip(conditionings, c_regions, strict=True):
                if r is None:
                    r = TextConditioningRegions(
                        masks=torch.ones((1, 1, h, w), dtype=dtype),
                        ranges=[Range(start=0, end=c.embeds.shape[1])],
                    )
                tmp_regions.append(r)

            if unet_kwargs.cross_attention_kwargs is None:
                unet_kwargs.cross_attention_kwargs = {}

            unet_kwargs.cross_attention_kwargs.update(
                regional_prompt_data=RegionalPromptData(regions=tmp_regions, device=device, dtype=dtype),
            )

    @staticmethod
    def _pad_zeros(t: torch.Tensor, pad_shape: tuple, dim: int) -> torch.Tensor:
        return torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=dim)

    @classmethod
    def _pad_conditioning(
        cls,
        cond: torch.Tensor,
        target_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad provided conditioning tensor to target_len by zeros and returns mask of unpadded bytes.

        Args:
            cond (torch.Tensor): Conditioning tensor which to pads by zeros.
            target_len (int): To which length(tokens count) pad tensor.
        """
        conditioning_attention_mask = torch.ones((cond.shape[0], cond.shape[1]), device=cond.device, dtype=cond.dtype)

        if cond.shape[1] < target_len:
            conditioning_attention_mask = cls._pad_zeros(
                conditioning_attention_mask,
                pad_shape=(cond.shape[0], target_len - cond.shape[1]),
                dim=1,
            )

            cond = cls._pad_zeros(
                cond,
                pad_shape=(cond.shape[0], target_len - cond.shape[1], cond.shape[2]),
                dim=1,
            )

        return cond, conditioning_attention_mask

    @classmethod
    def _concat_conditionings_for_batch(
        cls,
        conditionings: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Concatenate provided conditioning tensors to one batched tensor.
        If tensors have different sizes then pad them by zeros and creates
        encoder_attention_mask to exclude padding from attention.

        Args:
            conditionings (List[torch.Tensor]): List of conditioning tensors to concatenate.
        """
        encoder_attention_mask = None
        max_len = max([c.shape[1] for c in conditionings])
        if any(c.shape[1] != max_len for c in conditionings):
            encoder_attention_masks = [None] * len(conditionings)
            for i in range(len(conditionings)):
                conditionings[i], encoder_attention_masks[i] = cls._pad_conditioning(conditionings[i], max_len)
            encoder_attention_mask = torch.cat(encoder_attention_masks)

        return torch.cat(conditionings), encoder_attention_mask


class ScheduledTextConditioningData:
    """
    Text conditioning data that supports A1111-style prompt scheduling.
    Wraps multiple TextConditioningData objects, one for each schedule step.
    """

    def __init__(
        self,
        uncond_field_data: ConditioningFieldData,
        cond_field_data: ConditioningFieldData,
        uncond_regions: Optional[TextConditioningRegions],
        cond_regions: Optional[TextConditioningRegions],
        guidance_scale: Union[float, List[float]],
        guidance_rescale_multiplier: float = 0,
    ):
        self.uncond_field_data = uncond_field_data
        self.cond_field_data = cond_field_data
        self.uncond_regions = uncond_regions
        self.cond_regions = cond_regions
        self.guidance_scale = guidance_scale
        self.guidance_rescale_multiplier = guidance_rescale_multiplier

    def get_text_conditioning_for_step(self, step: int) -> TextConditioningData:
        """
        Get the TextConditioningData for a specific denoising step.

        Args:
            step: Current step (1-indexed)

        Returns:
            TextConditioningData configured for this step
        """
        uncond = self.uncond_field_data.get_conditioning_for_step(step)
        cond = self.cond_field_data.get_conditioning_for_step(step)

        return TextConditioningData(
            uncond_text=uncond,
            cond_text=cond,
            uncond_regions=self.uncond_regions,
            cond_regions=self.cond_regions,
            guidance_scale=self.guidance_scale,
            guidance_rescale_multiplier=self.guidance_rescale_multiplier,
        )

    @property
    def has_scheduling(self) -> bool:
        return self.uncond_field_data.has_scheduling or self.cond_field_data.has_scheduling
