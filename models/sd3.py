import math

import diffusers
import torch
from torch import nn
import torch.nn.functional as F

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE


KEEP_IN_HIGH_PRECISION = ['pos_embed', 'time_text_embed', 'context_embedder', 'norm_out', 'proj_out']


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


class SD3Pipeline(BasePipeline):
    name = 'sd3'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['JointTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        dtype = self.model_config['dtype']

        if diffusers_path := self.model_config.get('diffusers_path', None):
            self.diffusers_pipeline = diffusers.StableDiffusion3Pipeline.from_pretrained(diffusers_path, torch_dtype=dtype, transformer=None)
        else:
            raise NotImplementedError()

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)
        if diffusers_path := self.model_config.get('diffusers_path', None):
            transformer = diffusers.SD3Transformer2DModel.from_pretrained(diffusers_path, torch_dtype=dtype, subfolder='transformer')
        else:
            raise NotImplementedError()

        for name, p in transformer.named_parameters():
            if not (any(x in name for x in KEEP_IN_HIGH_PRECISION) or p.ndim == 1):
                p.data = p.data.to(transformer_dtype)

        self.diffusers_pipeline.transformer = transformer

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2, self.text_encoder_3]

    def save_adapter(self, save_dir, peft_state_dict):
        self.save_lora_weights(save_dir, transformer_lora_layers=peft_state_dict)

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        if text_encoder == self.text_encoder:
            def fn(caption, is_video):
                assert not any(is_video)
                prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                    prompt=caption,
                    device=text_encoder.device,
                    clip_model_index=0,
                )
                return {'prompt_embed': prompt_embed, 'pooled_prompt_embed': pooled_prompt_embed}
            return fn
        elif text_encoder == self.text_encoder_2:
            def fn(caption, is_video):
                assert not any(is_video)
                prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                    prompt=caption,
                    device=text_encoder.device,
                    clip_model_index=1,
                )
                return {'prompt_2_embed': prompt_2_embed, 'pooled_prompt_2_embed': pooled_prompt_2_embed}
            return fn
        elif text_encoder == self.text_encoder_3:
            def fn(caption, is_video):
                assert not any(is_video)
                prompt_3_embed, pooled_prompt_3_embed = self._get_clip_prompt_embeds(
                    prompt=caption,
                    device=text_encoder.device,
                    clip_model_index=2,
                )
                return {'prompt_3_embed': prompt_3_embed, 'pooled_prompt_3_embed': pooled_prompt_3_embed}
            return fn
        else:
            raise ValueError(f"Unknown text encoder: {text_encoder}")

    def prepare_inputs(self, inputs, timestep_quantile=None):
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
            layers.append(TransformerLayer(block))
        
        layers.append(FinalLayer(self))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        latents = inputs['latents']
        timestep = inputs.get('timestep', None)
        prompt_embed = inputs.get('prompt_embed', None)
        prompt_2_embed = inputs.get('prompt_2_embed', None)
        prompt_3_embed = inputs.get('prompt_3_embed', None)
        pooled_prompt_embed = inputs.get('pooled_prompt_embed', None)
        pooled_prompt_2_embed = inputs.get('pooled_prompt_2_embed', None)
        pooled_prompt_3_embed = inputs.get('pooled_prompt_3_embed', None)

        # Prepare timestep embeddings
        if timestep is not None:
            t_emb = self.model.transformer.time_embedding(timestep)
        else:
            t_emb = None

        # Prepare text embeddings
        if prompt_embed is not None:
            txt_emb = self.model.transformer.text_embedding(prompt_embed)
        else:
            txt_emb = None

        # Combine embeddings
        hidden_states = self.model.transformer.initial_layer(latents, t_emb, txt_emb)
        
        return {'hidden_states': hidden_states}


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states = inputs['hidden_states']
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