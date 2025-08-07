import math
import os.path
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/HiDream'))

import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.utils.logging import logger
from safetensors.torch import save_file
import transformers
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from einops import repeat
import diffusers

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, empty_cuda_cache
from utils.offloading import ModelOffloader
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.models.moe import MoEGate


KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 't_embedder', 'p_embedder', 'x_embedder', 'final_layer', 'gate']


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


class HiDreamPipeline(BasePipeline):
    name = 'hidream'

    checkpointable_layers = [
        'TransformerWrapper',
        'SingleTransformerWrapper',
    ]

    adapter_target_modules = ['HiDreamImageTransformerBlock', 'HiDreamImageSingleTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        dtype = self.model_config['dtype']
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        llama3_path = self.model_config['llama3_path']
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
            llama3_path,
            use_fast=False
        )
        self.diffusers_pipeline = HiDreamImagePipeline.from_pretrained(
            self.model_config['diffusers_path'],
            scheduler=scheduler,
            tokenizer_4=tokenizer_4,
            text_encoder_4=None,
            torch_dtype=dtype
        )

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        llama3_path = self.model_config['llama3_path']
        if self.model_config.get('llama3_4bit', False):
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quantization_config = None
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            llama3_path,
            output_hidden_states=True,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        for p in text_encoder_4.parameters():
            p.requires_grad_(False)
            p.data = p.data.to('cpu')
        empty_cuda_cache()
        self.diffusers_pipeline.text_encoder_4 = text_encoder_4

        if transformer_dtype == 'nf4':
            quantization_config = diffusers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quantization_config = None

        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            self.model_config['diffusers_path'],
            torch_dtype=dtype,
            subfolder='transformer',
            quantization_config=quantization_config,
        )

        for name, p in transformer.named_parameters():
            if not (any(x in name for x in KEEP_IN_HIGH_PRECISION) or p.ndim == 1):
                p.data = p.data.to(transformer_dtype)

        self.diffusers_pipeline.transformer = transformer

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder_4]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        if text_encoder == self.text_encoder_4:
            def fn(caption, is_video):
                # args are lists
                prompt_embeds = self._get_llama3_prompt_embeds(
                    prompt=caption,
                    device=text_encoder.device,
                )
                return {'prompt_embeds': prompt_embeds}
            return fn
        else:
            raise ValueError(f"Unknown text encoder: {text_encoder}")

    def _get_llama3_prompt_embeds(
        self,
        prompt,
        device=None,
        max_sequence_length=256,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.tokenizer_4(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_4.batch_decode(untruncated_ids[:, self.tokenizer_4.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_4.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder_4.config, "use_attention_mask") and self.text_encoder_4.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder_4(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

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
        
        # Add Llama layer for text processing
        layers.append(LlamaLayer(self.text_encoder_4))
        
        for i, block in enumerate(self.transformer.transformer_blocks):
            if hasattr(block, 'is_single'):
                layers.append(SingleTransformerWrapper(block, i, i, self.offloader_single))
            else:
                layers.append(TransformerWrapper(block, i, i, self.offloader_double))
        
        layers.append(OutputLayer(self))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        self.offloader_double = ModelOffloader(
            'hidream_double',
            blocks_to_swap,
            self.model_config.get('block_swap_buffer_size', 1024),
            self.model_config.get('block_swap_buffer_size', 1024),
            True,
            torch.device('cuda'),
            False,
            debug=False
        )
        self.offloader_single = ModelOffloader(
            'hidream_single',
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


class LlamaLayer(nn.Module):
    def __init__(self, llama_model):
        super().__init__()
        self.llama_model = llama_model

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        prompt_embeds = inputs.get('prompt_embeds', None)
        if prompt_embeds is not None:
            # Process through Llama model
            hidden_states = self.llama_model(prompt_embeds).last_hidden_state
            inputs['llama_hidden_states'] = hidden_states
        return inputs


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
        llama_hidden_states = inputs.get('llama_hidden_states', None)

        # Prepare timestep embeddings
        if timestep is not None:
            t_emb = self.model.transformer.time_embedding(timestep)
        else:
            t_emb = None

        # Prepare text embeddings
        if llama_hidden_states is not None:
            txt_emb = self.model.transformer.text_embedding(llama_hidden_states)
        else:
            txt_emb = None

        # Combine embeddings
        hidden_states = self.model.transformer.initial_layer(latents, t_emb, txt_emb)
        
        return {'hidden_states': hidden_states}


class TransformerWrapper(nn.Module):
    def __init__(self, block, block_idx, global_block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.global_block_idx = global_block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states = inputs['hidden_states']
        
        if self.offloader is not None:
            hidden_states = self.offloader.forward(self.block, hidden_states, self.global_block_idx)
        else:
            hidden_states = self.block(hidden_states)
        
        return {'hidden_states': hidden_states}


def concatenate_hidden_states(inputs):
    # Helper function to concatenate hidden states from different sources
    hidden_states = inputs['hidden_states']
    if 'additional_hidden_states' in inputs:
        additional = inputs['additional_hidden_states']
        hidden_states = torch.cat([hidden_states, additional], dim=-1)
    return hidden_states


class SingleTransformerWrapper(nn.Module):
    def __init__(self, block, block_idx, global_block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.global_block_idx = global_block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states = inputs['hidden_states']
        
        if self.offloader is not None:
            hidden_states = self.offloader.forward(self.block, hidden_states, self.global_block_idx)
        else:
            hidden_states = self.block(hidden_states)
        
        return {'hidden_states': hidden_states}


class OutputLayer(nn.Module):
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