from contextlib import nullcontext

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.stable_diffusion.vae_tiling import patch_vae_tiling_params
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_sd15_sdxl


# Constants matching redone comfyui vae defaults
DEFAULT_TILE_SIZE_LATENT = 64   # 512px output tiles
DEFAULT_OVERLAP_LATENT = 8      # 64px overlap
DOWNSCALE_RATIO = 8


class TiledVAEDecoder:
    """Memory-efficient tiled VAE decoder with CPU blending"""

    def __init__(self, vae, device, vae_dtype):
        self.vae = vae
        self.device = device
        self.vae_dtype = vae_dtype

    def _make_linear_ramp(self, size: int, direction: str = 'up') -> torch.Tensor:
        """Create a 1D linear ramp on CPU"""
        if direction == 'up':
            return torch.linspace(0.0, 1.0, size, dtype=torch.float32)
        else:
            return torch.linspace(1.0, 0.0, size, dtype=torch.float32)

    def _decode_tile_to_cpu(self, samples: torch.Tensor) -> torch.Tensor:
        """Decode a single tile and return result on CPU immediately"""
        # Move tile to GPU
        samples_gpu = samples.to(self.vae_dtype).to(self.device)

        # Decode
        decoded = self.vae.decode(samples_gpu / self.vae.config.scaling_factor, return_dict=False)[0]

        # Normalize, clamp, move to CPU immediately
        result = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0).cpu()

        # Explicitly delete GPU tensors
        del samples_gpu
        del decoded

        return result

    def decode(
        self,
        samples: torch.Tensor,
        tile_x: int = DEFAULT_TILE_SIZE_LATENT,
        tile_y: int = DEFAULT_TILE_SIZE_LATENT,
        overlap: int = DEFAULT_OVERLAP_LATENT,
        progress_callback=None
    ) -> torch.Tensor:
        """
        Tiled decode with CPU accumulation.
        GPU only processes one tile at a time.

        Args:
            samples: Latent tensor (can be on any device, will be moved to CPU)
            tile_x: Tile width in latent space (default 64 = 512px)
            tile_y: Tile height in latent space (default 64 = 512px)
            overlap: Overlap in latent space (default 8 = 64px)
        """
        # Move samples to CPU for processing
        samples = samples.cpu()

        b, c, h, w = samples.shape
        out_h = h * DOWNSCALE_RATIO
        out_w = w * DOWNSCALE_RATIO
        overlap_px = overlap * DOWNSCALE_RATIO

        # Clamp tile size to image size
        tile_x = min(tile_x, w)
        tile_y = min(tile_y, h)

        # Single tile case - no blending needed
        if w <= tile_x and h <= tile_y:
            return self._decode_tile_to_cpu(samples)

        # Clear GPU memory before starting
        if self.device.type != 'cpu':
            torch.cuda.empty_cache()

        # All accumulation happens on CPU - uses system RAM
        output = torch.zeros((b, 3, out_h, out_w), dtype=torch.float32)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

        # Calculate step size
        step_x = max(1, tile_x - overlap)
        step_y = max(1, tile_y - overlap)

        # Build tile list
        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x_end = min(x + tile_x, w)
                y_end = min(y + tile_y, h)
                tiles.append((x, y, x_end, y_end))
                x += step_x
            y += step_y

        # Process each tile
        for idx, (x, y, x_end, y_end) in enumerate(tiles):
            # Extract tile (on CPU)
            tile = samples[:, :, y:y_end, x:x_end]

            # Decode - GPU work happens here, result comes back to CPU
            decoded = self._decode_tile_to_cpu(tile)

            # Output coordinates in pixel space
            ox = x * DOWNSCALE_RATIO
            oy = y * DOWNSCALE_RATIO
            ox_end = x_end * DOWNSCALE_RATIO
            oy_end = y_end * DOWNSCALE_RATIO

            th = oy_end - oy
            tw = ox_end - ox

            # Build blend mask on CPU
            blend = torch.ones((1, 1, th, tw), dtype=torch.float32)

            # Left edge blend
            if x > 0 and overlap_px > 0:
                ramp_len = min(overlap_px, tw)
                ramp = self._make_linear_ramp(ramp_len, 'up')
                blend[:, :, :, :ramp_len] *= ramp.view(1, 1, 1, -1)

            # Right edge blend
            if x_end < w and overlap_px > 0:
                ramp_len = min(overlap_px, tw)
                ramp = self._make_linear_ramp(ramp_len, 'down')
                blend[:, :, :, -ramp_len:] *= ramp.view(1, 1, 1, -1)

            # Top edge blend
            if y > 0 and overlap_px > 0:
                ramp_len = min(overlap_px, th)
                ramp = self._make_linear_ramp(ramp_len, 'up')
                blend[:, :, :ramp_len, :] *= ramp.view(1, 1, -1, 1)

            # Bottom edge blend
            if y_end < h and overlap_px > 0:
                ramp_len = min(overlap_px, th)
                ramp = self._make_linear_ramp(ramp_len, 'down')
                blend[:, :, -ramp_len:, :] *= ramp.view(1, 1, -1, 1)

            # Accumulate on CPU
            output[:, :, oy:oy_end, ox:ox_end] += decoded * blend
            weights[:, :, oy:oy_end, ox:ox_end] += blend

            # Free decoded tensor
            del decoded
            del blend

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(tiles))

        # Normalize by accumulated weights
        result = output / weights.clamp(min=1e-8)

        # Free accumulation buffers
        del output
        del weights

        return result


@invocation(
    "l2i",
    title="Latents to Image - SD1.5, SDXL",
    tags=["latents", "image", "vae", "l2i"],
    category="latents",
    version="1.4.0",
)
class LatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    tile_size: int = InputField(default=512, multiple_of=8, description=FieldDescriptions.vae_tile_size)
    fp32: bool = InputField(default=False, description=FieldDescriptions.fp32)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        use_tiling = self.tiled or context.config.get().force_tiled_decode

        vae_info = context.models.load(self.vae.vae)
        assert isinstance(vae_info.model, (AutoencoderKL, AutoencoderTiny))

        estimated_working_memory = estimate_vae_working_memory_sd15_sdxl(
            operation="decode",
            image_tensor=latents,
            vae=vae_info.model,
            tile_size=self.tile_size if use_tiling else None,
            fp32=self.fp32,
        )

        with (
            SeamlessExt.static_patch_model(vae_info.model, self.vae.seamless_axes),
            vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae),
        ):
            context.util.signal_progress("Running VAE decoder")
            assert isinstance(vae, (AutoencoderKL, AutoencoderTiny))

            # Set VAE dtype
            vae_dtype = torch.float32 if self.fp32 else torch.float16
            vae.to(dtype=vae_dtype)

            # Get device
            device = TorchDevice.choose_torch_device()

            # Clear memory before starting
            TorchDevice.empty_cache()

            if use_tiling:
                context.util.signal_progress("VAE tiled decode (CPU blend)")

                # Calculate tile parameters in latent space
                tile_latent = self.tile_size // DOWNSCALE_RATIO
                overlap_latent = DEFAULT_OVERLAP_LATENT  # 8 = 64px overlap

                # Create decoder
                decoder = TiledVAEDecoder(vae, device, vae_dtype)

                # Progress callback
                def progress(current, total):
                    context.util.signal_progress(f"Decoding tile {current}/{total}")

                # Decode with CPU blending
                image_tensor = decoder.decode(
                    samples=latents,
                    tile_x=tile_latent,
                    tile_y=tile_latent,
                    overlap=overlap_latent,
                    progress_callback=progress
                )

                # Convert to PIL
                np_image = image_tensor.permute(0, 2, 3, 1).numpy()
                image = VaeImageProcessor.numpy_to_pil(np_image)[0]

                # Free tensor
                del image_tensor

            else:
                # Non-tiled decode - original path
                latents = latents.to(vae_dtype).to(device)

                with torch.inference_mode():
                    latents = latents / vae.config.scaling_factor
                    image_tensor = vae.decode(latents, return_dict=False)[0]
                    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                    np_image = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                    image = VaeImageProcessor.numpy_to_pil(np_image)[0]

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)
