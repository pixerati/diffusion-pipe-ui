import math
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/OmniGen2'))

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from einops import rearrange
from deepspeed.utils.logging import logger

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline as OriginalOmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2RotaryPosEmbed


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    return t


class OmniGen2Pipeline(BasePipeline):
    name = 'omnigen2'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['OmniGen2TransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        dtype = self.model_config['dtype']

        self.diffusers_pipeline = OriginalOmniGen2Pipeline.from_pretrained(self.model_config['diffusers_path'], torch_dtype=dtype, transformer=None)

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        if diffusers_path := self.model_config.get('diffusers_path', None):
            transformer = OmniGen2Transformer2DModel.from_pretrained(diffusers_path, torch_dtype=dtype, subfolder='transformer')
        else:
            raise NotImplementedError()

        self.diffusers_pipeline.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.mllm]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def _get_qwen2_prompt_embeds(
        self,
        prompt,
        device=None,
        max_sequence_length=256,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.processor.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.processor.tokenizer.batch_decode(untruncated_ids[:, self.processor.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.processor.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.mllm.config, "use_attention_mask") and self.mllm.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.mllm(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # args are lists
            prompt_embeds = self._get_qwen2_prompt_embeds(
                prompt=caption,
                device=text_encoder.device,
            )
            return {'prompt_embeds': prompt_embeds}
        return fn

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
        prompt_embeds = inputs.get('prompt_embeds', None)

        # Prepare timestep embeddings
        if timestep is not None:
            t_emb = self.model.transformer.time_embedding(timestep)
        else:
            t_emb = None

        # Prepare text embeddings
        if prompt_embeds is not None:
            txt_emb = self.model.transformer.text_embedding(prompt_embeds)
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