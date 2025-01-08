import toml
import argparse
import json
import os

from submodules.HunyuanVideo.hyvideo.utils.file_utils import save_videos_grid
from submodules.HunyuanVideo.hyvideo.config import parse_args
from submodules.HunyuanVideo.hyvideo.inference import HunyuanVideoSampler
from submodules.HunyuanVideo.hyvideo.constants import NEGATIVE_PROMPT


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')  
parser.add_argument('--lora-adapter-path', help='Path to LoRA model.')

args = parser.parse_args()

if __name__ == '__main__':
  
    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))
        
    model_type = config['model']['type']

    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')
    
    output_path = config['output_dir']
    sample_save_path = os.path.join(output_path, "samples")
    
    os.makedirs(sample_save_path, exist_ok=True)
    
    sample_video_size = config['sample']['video_size']
    sample_video_length = config['sample']['video_length']
    sample_infer_steps = config['sample']['infer_steps']
    sample_prompt = config['sample']['prompt']
    sample_flow_reverse = config['sample']['flow_reverse']
    sample_use_cpu_offload = config['sample']['use_cpu_offload']
    
    lora_adapter_path = args.lora_adapter_path
    
    model.load_adapter_weights(lora_adapter_path)
    
    model()
    
    