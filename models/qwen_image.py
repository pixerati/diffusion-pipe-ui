import json
from pathlib import Path
from typing import Union, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import diffusers
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import transformers

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, iterate_safetensors
from utils.offloading import ModelOffloader


KEEP_IN_HIGH_PRECISION = ['time_text_embed', 'img_in', 'txt_in', 'norm_out', 'proj_out']


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


# I copied this because it doesn't handle encoder_hidden_states_mask, which causes high loss values when there is a lot
# of padding. When (or if) they fix it upstream, I don't want the changes to break my workaround, which is to just set
# attention_mask.
class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        residual = hidden_states
        hidden_states = attn.norm(hidden_states)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-1)

        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection and residual
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states


class QwenImagePipeline(BasePipeline):
    name = 'qwen_image'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['QwenImageTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        dtype = self.model_config['dtype']

        if diffusers_path := self.model_config.get('diffusers_path', None):
            self.diffusers_pipeline = diffusers.StableDiffusionXLPipeline.from_pretrained(diffusers_path, torch_dtype=dtype, transformer=None)
        else:
            raise NotImplementedError()

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)
        if diffusers_path := self.model_config.get('diffusers_path', None):
            transformer = diffusers.Qwen2Transformer2DModel.from_pretrained(diffusers_path, torch_dtype=dtype, subfolder='transformer')
        else:
            raise NotImplementedError()

        for name, p in transformer.named_parameters():
            if not (any(x in name for x in KEEP_IN_HIGH_PRECISION) or p.ndim == 1):
                p.data = p.data.to(transformer_dtype)

        self.diffusers_pipeline.transformer = transformer

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2]

    def save_adapter(self, save_dir, peft_state_dict):
        self.save_lora_weights(save_dir, transformer_lora_layers=peft_state_dict)

    def save_model(self, save_dir, state_dict):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        def fn(path):
            return PreprocessMediaFile(path, self.model_config.get('image_size', 1024))
        return fn

    def get_call_vae_fn(self, vae):
        def fn(image):
            latents = vae.encode(image.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def _get_qwen_prompt_embeds(
        self,
        prompt,
        device=None,
        dtype=None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # args are lists
            prompt_embeds = self._get_qwen_prompt_embeds(
                prompt=caption,
                device=text_encoder.device,
                dtype=text_encoder.dtype,
            )
            return {'prompt_embeds': prompt_embeds}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        # Add image conditioning
        if 'image' in inputs:
            image = inputs['image']
            if image.ndim == 3:
                image = image.unsqueeze(0)
            inputs['image'] = image

        # Add timestep conditioning
        if timestep_quantile is not None:
            batch_size = inputs['latents'].shape[0]
            t = torch.full((batch_size,), timestep_quantile, device=inputs['latents'].device, dtype=inputs['latents'].dtype)
            inputs['timestep'] = t

        return inputs

    def to_layers(self):
        layers = []
        layers.append(InitialLayer(self))
        
        for i, block in enumerate(self.transformer.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        
        layers.append(FinalLayer(self))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        self.offloader = ModelOffloader(
            'qwen_image',
            blocks_to_swap,
            self.model_config.get('block_swap_buffer_size', 1024),
            self.model_config.get('block_swap_buffer_size', 1024),
            True,
            torch.device('cuda'),
            False,
            debug=False
        )

    def prepare_block_swap_training(self):
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if not disable_block_swap:
            self.transformer.eval()
            for name, p in self.transformer.named_parameters():
                p.original_name = name


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        latents = inputs['latents']
        timestep = inputs.get('timestep', None)
        prompt_embeds = inputs.get('prompt_embeds', None)
        image = inputs.get('image', None)

        # Prepare timestep embeddings
        if timestep is not None:
            t_emb = self.model.transformer.time_embedding(timestep)
        else:
            t_emb = None

        # Prepare image embeddings
        if image is not None:
            img_emb = self.model.transformer.image_embedding(image)
        else:
            img_emb = None

        # Prepare text embeddings
        if prompt_embeds is not None:
            txt_emb = self.model.transformer.text_embedding(prompt_embeds)
        else:
            txt_emb = None

        # Combine embeddings
        hidden_states = self.model.transformer.initial_layer(latents, t_emb, img_emb, txt_emb)
        
        return {'hidden_states': hidden_states}

    def rope_params(self, index, dim, theta=10000):
        # Generate rotary position embeddings
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(index, dtype=freqs.dtype)
        freqs = torch.outer(t, freqs)
        return torch.cat((freqs, freqs), dim=-1)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states = inputs['hidden_states']
        
        if self.offloader is not None:
            hidden_states = self.offloader.forward(self.block, hidden_states, self.block_idx)
        else:
            hidden_states = self.block(hidden_states)
        
        return {'hidden_states': hidden_states}


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states = inputs['hidden_states']
        
        # Final processing
        output = self.model.transformer.final_layer(hidden_states)
        
        return {'output': output} 