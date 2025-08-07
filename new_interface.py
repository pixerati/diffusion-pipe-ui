import queue
import signal
import subprocess
import threading
import gradio as gr
import os
from datetime import datetime, timedelta
import json
import torch
import deepspeed
from deepspeed import comm as dist
import torch.multiprocessing
from utils.patches import apply_patches
import multiprocess as mp
import argparse
import toml
import shutil
import zipfile
import tempfile
import time

# Working directories
MODEL_DIR = "/workspace/models"
BASE_DATASET_DIR = "/workspace/datasets"
OUTPUT_DIR = "/workspace/outputs"
CONFIG_DIR = "/workspace/configs"

# Maximum number of media to display in the gallery
MAX_MEDIA = 50

# Determine if running on Runpod by checking the environment variable
IS_RUNPOD = os.getenv("IS_RUNPOD", "false").lower() == "true"

CONDA_DIR = os.getenv("CONDA_DIR", "/opt/conda")  # Directory where Conda is installed in the Docker container

# Maximum upload size in MB (Gradio expects max_file_size in MB)
MAX_UPLOAD_SIZE_MB = 500 if IS_RUNPOD else None  # 500MB or no limit

process_dict = {}
process_lock = threading.Lock()

log_queue = queue.Queue()

# Define custom CSS for styling the log box and search box
custom_log_box_css = """
#log_box textarea {
    overflow-y: scroll;
    max-height: 400px;  /* Set a max height for the log box */
    white-space: pre-wrap;  /* Preserve line breaks and white spaces */
    border: 1px solid #ccc;
    padding: 10px;
    font-family: monospace;
    scrollbar-width: thin!important;
}

#file_explorer {
max-height: 374px!important;
}

#file_explorer .file-wrap {
max-height: 320px!important;
}
"""

def read_subprocess_output(proc, log_queue):
    for line in iter(proc.stdout.readline, b''):
        decoded_line = line.decode('utf-8')
        log_queue.put(decoded_line)
    proc.stdout.close()
    proc.wait()
    with process_lock:
        pid = proc.pid
        if pid in process_dict:
            del process_dict[pid]

def update_logs(log_box, subprocess_proc):
    new_logs = ""
    while not log_queue.empty():
        new_logs += log_queue.get()
    return log_box + new_logs

def clear_logs():
    while not log_queue.empty():
        log_queue.get()
    return ""

def create_dataset_config(dataset_path: str,
                        config_dir: str,
                        num_repeats: int, 
                        resolutions: list, 
                        enable_ar_bucket: bool, 
                        min_ar: float, 
                        max_ar: float, 
                        num_ar_buckets: int, 
                        frame_buckets: list,
                        ar_buckets: list) -> str:
    """Create and save the dataset configuration in TOML format."""
    dataset_config = {
        "resolutions": resolutions,
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": frame_buckets,
        "ar_buckets": ar_buckets,	
        "directory": [
            {
                "path": dataset_path,
                "num_repeats": num_repeats
            }
        ]
    }
    dataset_file = f"dataset_config.toml"
    dataset_path_full = os.path.join(config_dir, dataset_file)
    with open(dataset_path_full, "w") as f:
        toml.dump(dataset_config, f)
    return dataset_path_full

def create_training_config(
    # Main training parameters
    output_dir: str,
    config_dir: str,
    dataset_config_path: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
    warmup_steps: int,
    eval_every: int,
    eval_before_first_step: bool,
    eval_micro_batch_size_per_gpu: int,
    eval_gradient_accumulation_steps: int,
    save_every: int,
    checkpoint_every_n_minutes: int,
    activation_checkpointing: bool,
    partition_method: str,
    save_dtype: str,
    caching_batch_size: int,
    steps_per_print: int,
    video_clip_mode: str,
    
    # Model parameters
    model_type: str,
    transformer_path: str,
    vae_path: str,
    llm_path: str,
    clip_path: str,
    dtype: str = "float16",
    
    # Adapter parameters
    rank: int = 8,
    only_double_blocks: bool = False,
    
    # Optimizer parameters
    optimizer_type: str = "adamw_optimi",
    lr: float = 1e-4,
    betas: str = "(0.9, 0.999)",
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    
    # Monitoring parameters
    enable_wandb: bool = False,
    wandb_run_name: str = None,
    wandb_tracker_name: str = None,
    wandb_api_key: str = None,
):
    """
    Creates a training configuration dictionary from individual parameters.
    """
    
    num_gpus = int(os.getenv("NUM_GPUS", "1"))
    
    # Base configuration
    training_config = {
        "output_dir": output_dir,
        "dataset": dataset_config_path,
        "epochs": epochs,
        "micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "warmup_steps": warmup_steps,
        "eval_every_n_epochs": eval_every,
        "eval_before_first_step": eval_before_first_step,
        "eval_micro_batch_size_per_gpu": eval_micro_batch_size_per_gpu,
        "eval_gradient_accumulation_steps": eval_gradient_accumulation_steps,
        "save_every_n_epochs": save_every,
        "checkpoint_every_n_minutes": checkpoint_every_n_minutes,
        "activation_checkpointing": activation_checkpointing,
        "partition_method": partition_method,
        "save_dtype": save_dtype,
        "caching_batch_size": caching_batch_size,
        "steps_per_print": steps_per_print,
        "video_clip_mode": video_clip_mode,
        "pipeline_stages": num_gpus,
    }
    
    # Model configuration based on model type
    if model_type == "hunyuan-video":
        training_config["model"] = {
            "type": "hunyuan-video",
            "transformer_path": transformer_path,
            "vae_path": vae_path,
            "llm_path": llm_path,
            "clip_path": clip_path,
            "dtype": dtype,
            "transformer_dtype": "float8",
            "timestep_sample_method": "logit_normal"
        }
    elif model_type == "wan":
        training_config["model"] = {
            "type": "wan",
            "ckpt_path": transformer_path,
            "dtype": dtype,
            "timestep_sample_method": "logit_normal"
        }
    
    # Adapter configuration with fixed type
    training_config["adapter"] = {
        "type": "lora",
        "rank": rank,
        "dtype": dtype,
        "only_double_blocks": only_double_blocks
    }
    
    # Optimizer configuration with evaluated betas
    training_config["optimizer"] = {
        "type": optimizer_type,
        "lr": lr,
        "betas": eval(betas),
        "weight_decay": weight_decay,
        "eps": eps
    }
    
    training_config["monitoring"] = {
        "log_dir": output_dir,
        "enable_wandb": enable_wandb,
        "wandb_run_name": wandb_run_name,
        "wandb_tracker_name": wandb_tracker_name,
        "wandb_api_key": wandb_api_key
    }
    
    training_file = f"training_config.toml"
    training_path_full = os.path.join(config_dir, training_file)
    with open(training_path_full, "w") as f:
        toml.dump(training_config, f)
    return (training_path_full, training_config)

def get_datasets():
    datasets = []
    for dataset in os.listdir(BASE_DATASET_DIR):
        datasets.append(dataset)
    return datasets

def load_training_config(dataset_name):
    training_config_path = os.path.join(CONFIG_DIR, dataset_name, "training_config.toml")
    dataset_config_path = os.path.join(CONFIG_DIR, dataset_name, "dataset_config.toml")
    
    config = {}
    
    # Load training configuration
    if not os.path.exists(training_config_path):
        return None, f"Training configuration file not found for the dataset '{dataset_name}'."
    
    try:
        with open(training_config_path, "r") as f:
            training_config = toml.load(f)
        config.update(training_config)
    except Exception as e:
        return None, f"Error loading training configuration: {str(e)}"
    
    # Load dataset configuration
    if not os.path.exists(dataset_config_path):
        return None, f"Dataset configuration file not found for the dataset '{dataset_name}'."
    
    try:
        with open(dataset_config_path, "r") as f:
            dataset_config = toml.load(f)
        config["dataset"] = dataset_config
    except Exception as e:
        return None, f"Error loading dataset configuration: {str(e)}"
    
    return config, None

def extract_config_values(config):
    """
    Extracts training parameters from the configuration dictionary.

    Args:
        config (dict): Dictionary containing training configurations.

    Returns:
        dict: Dictionary with the extracted values.
    """
    training_params = config.get("epochs", 1000)
    batch_size = config.get("micro_batch_size_per_gpu", 1)
    lr = config.get("optimizer", {}).get("lr", 2e-5)
    save_every = config.get("save_every_n_epochs", 2)
    eval_every = config.get("eval_every_n_epochs", 1)
    rank = config.get("adapter", {}).get("rank", 32)
    only_double_blocks = config.get("adapter", {}).get("only_double_blocks", False)
    dtype = config.get("adapter", {}).get("dtype", "bfloat16")
    
    # Get model type and paths based on model type
    model_type = config.get("model", {}).get("type", "hunyuan-video")
    
    if model_type == "hunyuan-video":
        transformer_path = config.get("model", {}).get("transformer_path", "/workspace/models/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors")
        vae_path = config.get("model", {}).get("vae_path", "/workspace/models/hunyuan_video_vae_fp32.safetensors")
        llm_path = config.get("model", {}).get("llm_path", "/workspace/models/llava-llama-3-8b-text-encoder-tokenizer")
        clip_path = config.get("model", {}).get("clip_path", "/workspace/models/clip-vit-large-patch14")
    elif model_type == "wan":
        transformer_path = config.get("model", {}).get("ckpt_path", "/data2/imagegen_models/Wan2.1-T2V-1.3B")
        vae_path = ""
        llm_path = ""
        clip_path = ""
    else:
        transformer_path = ""
        vae_path = ""
        llm_path = ""
        clip_path = ""
    
    optimizer_type = config.get("optimizer", {}).get("type", "adamw_optimi")
    betas = config.get("optimizer", {}).get("betas", [0.9, 0.99])
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.01)
    eps = config.get("optimizer", {}).get("eps", 1e-8)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    num_repeats = config.get('dataset', {}).get('directory', [{}])[:1][0].get('num_repeats', 10)
    resolutions = config.get("dataset", {}).get("resolutions", [512])
    enable_ar_bucket = config.get("dataset", {}).get("enable_ar_bucket", True)
    min_ar = config.get("dataset", {}).get("min_ar", 0.5)
    max_ar = config.get("dataset", {}).get("max_ar", 2.0)
    num_ar_buckets = config.get("dataset", {}).get("num_ar_buckets", 7)
    ar_buckets = config.get("dataset", {}).get("ar_buckets", None)
    frame_buckets = config.get("dataset", {}).get("frame_buckets", [1, 33, 65])
    gradient_clipping = config.get("gradient_clipping", 1.0)
    warmup_steps = config.get("warmup_steps", 100)
    eval_before_first_step = config.get("eval_before_first_step", True)
    eval_micro_batch_size_per_gpu = config.get("eval_micro_batch_size_per_gpu", 1)
    eval_gradient_accumulation_steps = config.get("eval_gradient_accumulation_steps", 1)
    checkpoint_every_n_minutes = config.get("checkpoint_every_n_minutes", 120)
    activation_checkpointing = config.get("activation_checkpointing", True)
    partition_method = config.get("partition_method", "parameters")
    save_dtype = config.get("save_dtype", "bfloat16")
    caching_batch_size = config.get("caching_batch_size", 1)
    steps_per_print = config.get("steps_per_print", 1)
    video_clip_mode = config.get("video_clip_mode", "single_middle")
    
    # Convert lists to JSON strings to fill text fields
    betas_str = json.dumps(betas)
    resolutions_str = json.dumps(resolutions)
    frame_buckets_str = json.dumps(frame_buckets)
    ar_buckets_str = json.dumps(ar_buckets) if ar_buckets else ""
    
    wandb_enabled = config.get("monitoring", {}).get("enable_wandb", False)
    wandb_run_name = config.get("monitoring", {}).get("wandb_run_name", None)
    wandb_tracker_name = config.get("monitoring", {}).get("wandb_tracker_name", None)
    wandb_api_key = config.get("monitoring", {}).get("wandb_api_key", None)
    
    return {
        "model_type": model_type,
        "epochs": training_params,
        "batch_size": batch_size,
        "lr": lr,
        "save_every": save_every,
        "eval_every": eval_every,
        "rank": rank,
        "only_double_blocks": only_double_blocks,
        "dtype": dtype,
        "transformer_path": transformer_path,
        "vae_path": vae_path,
        "llm_path": llm_path,
        "clip_path": clip_path,
        "optimizer_type": optimizer_type,
        "betas": betas_str,
        "weight_decay": weight_decay,
        "eps": eps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_repeats": num_repeats,
        "resolutions_input": resolutions_str,
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "ar_buckets": ar_buckets_str,
        "frame_buckets": frame_buckets_str,
        "gradient_clipping": gradient_clipping,
        "warmup_steps": warmup_steps,
        "eval_before_first_step": eval_before_first_step,
        "eval_micro_batch_size_per_gpu": eval_micro_batch_size_per_gpu,
        "eval_gradient_accumulation_steps": eval_gradient_accumulation_steps,
        "checkpoint_every_n_minutes": checkpoint_every_n_minutes,
        "activation_checkpointing": activation_checkpointing,
        "partition_method": partition_method,
        "save_dtype": save_dtype,
        "caching_batch_size": caching_batch_size,
        "steps_per_print": steps_per_print,
        "video_clip_mode": video_clip_mode,
        "enable_wandb": wandb_enabled,
        "wandb_run_name": wandb_run_name,
        "wandb_tracker_name": wandb_tracker_name,
        "wandb_api_key": wandb_api_key
    }

def validate_resolutions(resolutions):
    try:
        # Attempt to parse the input as JSON
        resolutions_list = json.loads(resolutions)
        
        # Check if the parsed object is a list
        if not isinstance(resolutions_list, list):
            return "Error: resolutions must be a list.", None
        
        # Case 1: List of numbers (int or float)
        if all(isinstance(b, (int, float)) for b in resolutions_list):
            return None, resolutions_list
        
        # Case 2: List of lists of numbers
        elif all(
            isinstance(sublist, list) and all(isinstance(item, (int, float)) for item in sublist)
            for sublist in resolutions_list
        ):
            return None, resolutions_list
        
        else:
            return (
                "Error: resolutions must be a list of numbers or a list of lists of numbers. "
                "Valid examples: [512] or [512, 768, 1024] or [[512, 512], [1280, 720]]"
            ), None
    
    except json.JSONDecodeError as e:
        return f"Error parsing resolutions: {str(e)}", None
    except Exception as e:
        return f"Unexpected error while validating resolutions: {str(e)}", None
    
def validate_ar_buckets(ar_buckets):
    try:
        # Attempt to parse the input as JSON
        ar_buckets_list = json.loads(ar_buckets)
        
        # Check if the parsed object is a list
        if not isinstance(ar_buckets_list, list):
            return "Error: ar_buckets must be a list.", None
        
        # Case 1: List of numbers (int or float)
        if all(isinstance(b, (int, float)) for b in ar_buckets_list):
            return None, ar_buckets_list
        
        # Case 2: List of lists of numbers
        elif all(
            isinstance(sublist, list) and all(isinstance(item, (int, float)) for item in sublist)
            for sublist in ar_buckets_list
        ):
            return None, ar_buckets_list
        
        else:
            return (
                "Error: ar_buckets must be a list of numbers or a list of lists of numbers. "
                "Valid examples: [1.0, 1.5] or [[512, 512], [448, 576]]"
            ), None
    
    except json.JSONDecodeError as e:
        return f"Error parsing ar_buckets: {str(e)}", None
    except Exception as e:
        return f"Unexpected error while validating ar_buckets: {str(e)}", None
    
def toggle_dataset_option(option):
    if option == "Create New Dataset":
        # Show creation container and hide selection container
        return (
            gr.update(visible=True),    # Show create_new_container
            gr.update(visible=False),   # Hide select_existing_container
            gr.update(choices=[], value=None),      # Clear Dropdown of existing datasets
            gr.update(value=""),        # Clear Dataset Name
            gr.update(value=""),         # Clear Upload Status
            gr.update(value=""),         # Clear Dataset Path
            gr.update(visible=True),     # Hide Create Dataset Button
            gr.update(visible=True),     # Show Upload Files Button 
        )
    else:
        # Hide creation container and show selection container
        datasets = get_datasets()
        return (
            gr.update(visible=False),   # Hide create_new_container
            gr.update(visible=True),    # Show select_existing_container
            gr.update(choices=datasets if datasets else [], value=None),  # Update Dropdown
            gr.update(value=""),        # Clear Dataset Name
            gr.update(value=""),         # Clear Upload Status
            gr.update(value=""),         # Clear Dataset Path
            gr.update(visible=False),     # Show Create Dataset Button
            gr.update(visible=False),     # Hide Upload Files Button
        )

def train_model(dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
                gradient_accumulation_steps, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, ar_buckets, gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print, video_clip_mode, resume_from_checkpoint, only_double_blocks, enable_wandb, wandb_run_name, wandb_tracker_name, wandb_api_key, model_type
                ):
    try:
        # Validate inputs
        if not dataset_path or not os.path.exists(dataset_path) or dataset_path == BASE_DATASET_DIR:
            return "Error: Please provide a valid dataset path", None
        
        os.makedirs(config_dir, exist_ok=True)

        if not config_dir or not os.path.exists(config_dir) or config_dir == CONFIG_DIR:
            return "Error: Please provide a valid config path", None

        os.makedirs(output_dir, exist_ok=True)
        
        if not output_dir or not os.path.exists(output_dir) or output_dir == OUTPUT_DIR:
            return "Error: Please provide a valid output path", None
        
        resolutions_error, resolutions_list = validate_resolutions(resolutions)
        if resolutions_error:
            return resolutions_error, None
            
        try:
            frame_buckets_list = json.loads(frame_buckets)
            if not isinstance(frame_buckets_list, list) or not all(isinstance(b, int) for b in frame_buckets_list):
                return "Error: Frame buckets must be a list of integers. Example: [1, 33, 65]", None
        except Exception as e:
            return f"Error parsing frame buckets: {str(e)}", None
        
        ar_buckets_list = None
        
        if len(ar_buckets) > 0:
            ar_buckets_error, ar_buckets_list = validate_ar_buckets(ar_buckets)
            if ar_buckets_error:
                return ar_buckets_error, None
            
        if enable_wandb and (wandb_api_key is None or wandb_api_key == ""):
                return "Error: Wandb is enabled but API KEY is required.", None

        # Create configurations
        dataset_config_path = create_dataset_config(
            dataset_path=dataset_path,
            config_dir=config_dir,
            num_repeats=num_repeats,
            resolutions=resolutions_list,
            enable_ar_bucket=enable_ar_bucket,
            min_ar=min_ar,
            max_ar=max_ar,
            num_ar_buckets=num_ar_buckets,
            frame_buckets=frame_buckets_list,
            ar_buckets=ar_buckets_list
        )
        
        training_config_path, _ = create_training_config(
            output_dir=output_dir,
            config_dir=config_dir,
            dataset_config_path=dataset_config_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_every=save_every,
            eval_every=eval_every,
            rank=rank,
            only_double_blocks=only_double_blocks,
            dtype=dtype,
            transformer_path=transformer_path,
            vae_path=vae_path,
            llm_path=llm_path,
            clip_path=clip_path,
            optimizer_type=optimizer_type,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            warmup_steps=warmup_steps,
            eval_before_first_step=eval_before_first_step,
            eval_micro_batch_size_per_gpu=eval_micro_batch_size_per_gpu,
            eval_gradient_accumulation_steps=eval_gradient_accumulation_steps,
            checkpoint_every_n_minutes=checkpoint_every_n_minutes,
            activation_checkpointing=activation_checkpointing,
            partition_method=partition_method,
            save_dtype=save_dtype,
            caching_batch_size=caching_batch_size,
            steps_per_print=steps_per_print,
            video_clip_mode=video_clip_mode,
            enable_wandb=enable_wandb,
            wandb_run_name=wandb_run_name,
            wandb_tracker_name=wandb_tracker_name,
            wandb_api_key=wandb_api_key,
            model_type=model_type
        )

        conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
        conda_env_name = "pyenv"
        num_gpus = os.getenv("NUM_GPUS", "1")
        
        if not os.path.isfile(conda_activate_path):
            return "Error: Conda activation script not found", None
        
        resume_checkpoint =  "--resume_from_checkpoint" if resume_from_checkpoint else ""
        
        cmd = (
            f"bash -c 'source {conda_activate_path} && "
            f"conda activate {conda_env_name} && "
            f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 {'NCCL_SHM_DISABLE=1' if int(num_gpus) > 1 else ''} deepspeed --num_gpus={num_gpus} "
            f"train.py --deepspeed --config {training_config_path} {resume_checkpoint}'"          
        )
        
        # --regenerate_cache
            
        proc = subprocess.Popen(
            cmd,
            shell=True,  # Required for complex shell commands
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            universal_newlines=False  # To handle bytes
        )
        
        with process_lock:
            process_dict[proc.pid] = proc  
        
        thread = threading.Thread(target=read_subprocess_output, args=(proc, log_queue))
        thread.start()
        
        pid = proc.pid
        
        return "Training started! Logs will appear below.\n", pid

    except Exception as e:
        return f"Error during training: {str(e)}", None

def stop_training(pid):
    if pid is None:
        return "No training process is currently running."

    with process_lock:
        proc = process_dict.get(pid)

    if proc is None:
        return "No training process is currently running."

    if proc.poll() is not None:
        return "Training process has already finished."

    try:
        # Send SIGTERM signal to the entire process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)  # Wait 5 seconds to terminate
            with process_lock:
                del process_dict[pid]
            return "Training process terminated gracefully."
        except subprocess.TimeoutExpired:
            # Force termination if SIGTERM does not work
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            with process_lock:
                del process_dict[pid]
            return "Training process killed forcefully."
    except Exception as e:
        return f"Error stopping training process: {str(e)}"

def upload_dataset(files, current_dataset, action, dataset_name=None):
    """
    Handle uploaded dataset files and store them in a unique directory.
    Action can be 'start' (initialize a new dataset) or 'add' (add files to current dataset).
    """
    if action == "start":
        if not dataset_name:
            return current_dataset, "Please provide a dataset name.", []
        # Clean and format the dataset name
        dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        dataset_name = dataset_name.replace(" ", "_")  # Replace spaces with underscores
        dataset_dir = os.path.join(BASE_DATASET_DIR, dataset_name)
        if os.path.exists(dataset_dir):
            return current_dataset, f"Dataset '{dataset_name}' already exists. Please choose a different name.", []
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir, f"Started new dataset: {dataset_dir}", show_media(dataset_dir)

    if not current_dataset:
        return current_dataset, "Please start a new dataset before uploading files.", []
    
    if not files:
        return current_dataset, "No files selected for upload.", show_media(current_dataset)
    
    # Process uploaded files
    for file in files:
        try:
            # Get the file extension
            _, ext = os.path.splitext(file.name)
            ext = ext.lower()
            
            # Check if it's a supported media file
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov']:
                # Copy the file to the dataset directory
                shutil.copy(file.name, current_dataset)
                
                # If it's a media file, check for a corresponding caption file
                base_name = os.path.basename(file.name)
                caption_file = os.path.join(os.path.dirname(file.name), os.path.splitext(base_name)[0] + '.txt')
                
                # If caption file exists, copy it too
                if os.path.exists(caption_file):
                    shutil.copy(caption_file, current_dataset)
            elif ext == '.txt':
                # It's a caption file, copy it
                shutil.copy(file.name, current_dataset)
            elif ext == '.zip':
                # Extract zip file
                with zipfile.ZipFile(file.name, 'r') as zip_ref:
                    # Create a temporary directory for extraction
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        zip_ref.extractall(tmpdirname)
                        
                        # Copy all supported files from the extracted content
                        for root, _, extracted_files in os.walk(tmpdirname):
                            for extracted_file in extracted_files:
                                _, file_ext = os.path.splitext(extracted_file)
                                file_ext = file_ext.lower()
                                
                                if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov', '.txt']:
                                    src_path = os.path.join(root, extracted_file)
                                    shutil.copy(src_path, current_dataset)
            else:
                print(f"Skipping unsupported file: {file.name}")
        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")
    
    return current_dataset, f"Files uploaded to: {current_dataset}", show_media(current_dataset)

def show_media(dataset_path):
    """
    Get a list of media files from the dataset directory to display in the gallery.
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return []
    
    media_files = []
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file)
            ext = ext.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov']:
                media_files.append(file_path)
    
    # Limit the number of files to display
    return media_files[:MAX_MEDIA]

def get_selected_file(file_path):
    """
    Handle file selection from the file explorer.
    """
    if not file_path:
        return None, "No file selected."
    
    try:
        return file_path, f"Selected file: {os.path.basename(file_path)}"
    except Exception as e:
        return None, f"Error selecting file: {str(e)}"

def handle_download(dataset_path, download_options):
    """
    Create a ZIP file containing the selected dataset files.
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return None, "No dataset selected or dataset path does not exist."
    
    try:
        dataset_name = os.path.basename(dataset_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{dataset_name}_{timestamp}.zip"
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add dataset files
            if "Dataset" in download_options:
                for root, _, files in os.walk(dataset_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join("dataset", os.path.relpath(file_path, dataset_path))
                        zipf.write(file_path, arcname)
            
            # Add output files
            if "Outputs" in download_options:
                output_path = os.path.join(OUTPUT_DIR, dataset_name)
                if os.path.exists(output_path):
                    for root, _, files in os.walk(output_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.join("outputs", os.path.relpath(file_path, output_path))
                            zipf.write(file_path, arcname)
            
            # Add config files
            if "Configs" in download_options:
                config_path = os.path.join(CONFIG_DIR, dataset_name)
                if os.path.exists(config_path):
                    for root, _, files in os.walk(config_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.join("configs", os.path.relpath(file_path, config_path))
                            zipf.write(file_path, arcname)
        
        return zip_path, f"Download ready: {zip_filename}"
    
    except Exception as e:
        return None, f"Error creating download: {str(e)}"

def force_save(output_dir, save_type="save"):
    """
    Create a file in the output directory to signal the training process to save a checkpoint.
    """
    if not output_dir or not os.path.exists(output_dir):
        return "Error: Output directory does not exist."
    
    try:
        # Create a file to signal the training process
        signal_file = os.path.join(output_dir, f"{save_type}_signal.txt")
        with open(signal_file, "w") as f:
            f.write(f"Save requested at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return f"Signal sent to {save_type} the model."
    except Exception as e:
        return f"Error sending {save_type} signal: {str(e)}"

def update_ui_with_config(config_values):
    """
    Update UI components with values from the configuration.
    """
    if not config_values:
        return [gr.update(value=None) for _ in range(38)]  # Return empty updates for all fields
    
    return [
        config_values.get("epochs", 1000),
        config_values.get("batch_size", 1),
        config_values.get("lr", 2e-5),
        config_values.get("save_every", 2),
        config_values.get("eval_every", 1),
        config_values.get("rank", 32),
        config_values.get("only_double_blocks", False),
        config_values.get("dtype", "bfloat16"),
        config_values.get("transformer_path", ""),
        config_values.get("vae_path", ""),
        config_values.get("llm_path", ""),
        config_values.get("clip_path", ""),
        config_values.get("optimizer_type", "adamw_optimi"),
        config_values.get("betas", "[0.9, 0.99]"),
        config_values.get("weight_decay", 0.01),
        config_values.get("eps", 1e-8),
        config_values.get("gradient_accumulation_steps", 4),
        config_values.get("num_repeats", 10),
        config_values.get("resolutions_input", "[512]"),
        config_values.get("enable_ar_bucket", True),
        config_values.get("min_ar", 0.5),
        config_values.get("max_ar", 2.0),
        config_values.get("num_ar_buckets", 7),
        config_values.get("ar_buckets", ""),
        config_values.get("frame_buckets", "[1, 33]"),
        config_values.get("gradient_clipping", 1.0),
        config_values.get("warmup_steps", 100),
        config_values.get("eval_before_first_step", True),
        config_values.get("eval_micro_batch_size_per_gpu", 1),
        config_values.get("eval_gradient_accumulation_steps", 1),
        config_values.get("checkpoint_every_n_minutes", 120),
        config_values.get("activation_checkpointing", True),
        config_values.get("partition_method", "parameters"),
        config_values.get("save_dtype", "bfloat16"),
        config_values.get("caching_batch_size", 1),
        config_values.get("steps_per_print", 1),
        config_values.get("video_clip_mode", "single_middle"),
        config_values.get("enable_wandb", False),
        config_values.get("wandb_run_name", ""),
        config_values.get("wandb_tracker_name", ""),
        config_values.get("wandb_api_key", ""),
        config_values.get("model_type", "hunyuan-video")
    ]

# Gradio Interface
training_process_pid = gr.State(None)

with gr.Blocks(css=custom_log_box_css) as demo:
    gr.Markdown("# LoRA Training Interface for Video Generation Models")
    
    gr.Markdown("### Step 1: Dataset\nCreate a new dataset or select an existing one.")
    
    with gr.Row():
        dataset_option = gr.Radio(
            choices=["Create New Dataset", "Select Existing Dataset"],
            value="Create New Dataset",
            label="Dataset Option"
        )
    
    # Container for creating a new dataset
    with gr.Column(visible=True) as create_new_container:
        with gr.Row():
            dataset_name_input = gr.Textbox(
                label="Dataset Name",
                placeholder="Enter a name for your new dataset"
            )
            create_dataset_button = gr.Button("Create Dataset", visible=True)
        
        with gr.Row():
            upload_files = gr.File(
                label="Upload Files",
                file_count="multiple",
                file_types=["image", "video", "text", ".zip"],
                max_file_size=MAX_UPLOAD_SIZE_MB,
                visible=True
            )
    
    # Container for selecting an existing dataset
    with gr.Column(visible=False) as select_existing_container:
        existing_datasets = gr.Dropdown(
            label="Select Dataset",
            choices=get_datasets(),
            interactive=True
        )
    
    with gr.Row():
        upload_status = gr.Textbox(
            label="Status",
            interactive=False
        )
    
    # State to track the current dataset
    current_dataset_state = gr.State("")
    
    # Gallery to display dataset files
    gallery = gr.Gallery(
        label="Dataset Files",
        show_label=True,
        columns=5,
        object_fit="contain",
        height="auto"
    )
    
    # Handle dataset creation
    def handle_start_dataset(dataset_name):
        if not dataset_name:
            return "Please provide a dataset name.", "", gr.update(visible=True), gr.update(visible=True), ""
        
        # Clean and format the dataset name
        dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        dataset_name = dataset_name.replace(" ", "_")  # Replace spaces with underscores
        
        dataset_dir = os.path.join(BASE_DATASET_DIR, dataset_name)
        if os.path.exists(dataset_dir):
            return f"Dataset '{dataset_name}' already exists. Please choose a different name.", "", gr.update(visible=True), gr.update(visible=True), ""
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create config directory for this dataset
        config_dir = os.path.join(CONFIG_DIR, dataset_name)
        os.makedirs(config_dir, exist_ok=True)
        
        # Create output directory for this dataset
        output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create default dataset configuration
        try:
            create_dataset_config(
                dataset_path=dataset_dir,
                config_dir=config_dir,
                num_repeats=10,
                resolutions=[512],
                enable_ar_bucket=True,
                min_ar=0.5,
                max_ar=2.0,
                num_ar_buckets=7,
                frame_buckets=[1, 33, 65],
                ar_buckets=None
            )
            
            # Create default training configuration
            create_training_config(
                output_dir=output_dir,
                config_dir=config_dir,
                dataset_config_path=os.path.join(config_dir, "dataset_config.toml"),
                epochs=1000,
                batch_size=1,
                gradient_accumulation_steps=4,
                gradient_clipping=1.0,
                warmup_steps=100,
                eval_every=1,
                eval_before_first_step=True,
                eval_micro_batch_size_per_gpu=1,
                eval_gradient_accumulation_steps=1,
                save_every=2,
                checkpoint_every_n_minutes=120,
                activation_checkpointing=True,
                partition_method="parameters",
                save_dtype="bfloat16",
                caching_batch_size=1,
                steps_per_print=1,
                video_clip_mode="single_middle",
                model_type="hunyuan-video",
                transformer_path="/workspace/models/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors",
                vae_path="/workspace/models/hunyuan_video_vae_fp32.safetensors",
                llm_path="/workspace/models/llava-llama-3-8b-text-encoder-tokenizer",
                clip_path="/workspace/models/clip-vit-large-patch14",
                dtype="bfloat16",
                rank=32,
                only_double_blocks=False,
                optimizer_type="adamw_optimi",
                lr=2e-5,
                betas="[0.9, 0.99]",
                weight_decay=0.01,
                eps=1e-8,
                enable_wandb=False,
                wandb_run_name=None,
                wandb_tracker_name=None,
                wandb_api_key=None
            )
        except Exception as e:
            return f"Error creating default configuration: {str(e)}", "", gr.update(visible=True), gr.update(visible=True), ""
        
        return f"Created new dataset: {dataset_dir}", dataset_dir, gr.update(visible=False), gr.update(visible=True), dataset_dir
    
    # Handle file uploads
    upload_files.upload(
        fn=upload_dataset,
        inputs=[upload_files, current_dataset_state, gr.Textbox(value="add")],
        outputs=[current_dataset_state, upload_status, gallery]
    )
    
    # Handle selecting an existing dataset
    def handle_select_existing(selected_dataset):
        if not selected_dataset:
            return "", "", "", "No dataset selected.", [], gr.update(value=""), {}
        
        dataset_path = os.path.join(BASE_DATASET_DIR, selected_dataset)
        if not os.path.exists(dataset_path):
            return "", "", "", f"Dataset path does not exist: {dataset_path}", [], gr.update(value=""), {}
        
        # Load configuration if it exists
        config, error = load_training_config(selected_dataset)
        if error:
            return (
                dataset_path,  # Update dataset_path
                "",            # Don't update config_dir
                "",            # Don't update output_dir
                f"Error loading configuration: {error}",
                [],    # Clear gallery
                gr.update(value=""),         # Clear download status
                {}
            )
        config_values = extract_config_values(config)
        
        # Update config and output paths
        config_path = os.path.join(CONFIG_DIR, selected_dataset)
        output_path = os.path.join(OUTPUT_DIR, selected_dataset)
        
        return (
            dataset_path,  # Update dataset_path
            config_path,   # Update config_dir
            output_path,   # Update output_dir
            "",            # Clear error messages
            show_media(dataset_path),  # Update gallery with dataset files
            gr.update(value=""),        # Clear download status
            config_values  # Update training parameters
        )

    with gr.Row():
        with gr.Column():
            dataset_path = gr.Textbox(
                label="Dataset Path",
                value=BASE_DATASET_DIR,
                interactive=False
            )
            config_dir = gr.Textbox(
                label="Config Path",
                value=CONFIG_DIR,
                interactive=False
            )
            output_dir = gr.Textbox(
                label="Output Path",
                value=OUTPUT_DIR,
                interactive=False
            )
    
    create_dataset_button.click(
        fn=handle_start_dataset,
        inputs=dataset_name_input,
        outputs=[upload_status, current_dataset_state, create_dataset_button, upload_files, dataset_path]
    )
   
    dataset_option.change(
        fn=toggle_dataset_option,
        inputs=dataset_option,
        outputs=[create_new_container, select_existing_container, existing_datasets, dataset_name_input, upload_status, dataset_path, create_dataset_button, upload_files]
    )
    
    # Update config path and output path
    def update_config_output_path(dataset_path):
        config_path = os.path.join(CONFIG_DIR, os.path.basename(dataset_path))
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(dataset_path))
        return config_path, output_path
    
    # Update gallery when dataset path changes
    dataset_path.change(
        fn=lambda path: show_media(path),
        inputs=dataset_path,
        outputs=gallery
    )
    
    dataset_path.change(
        fn=update_config_output_path,
        inputs=dataset_path,
        outputs=[config_dir, output_dir]
    )
    
    # Model selection
    gr.Markdown("### Model Selection")
    with gr.Row():
        model_type = gr.Radio(
            choices=["hunyuan-video", "wan"],
            value="hunyuan-video",
            label="Model Type",
            info="Select the model type for training"
        )
    
    # Handle Models Configurations
    gr.Markdown("#### Models Configurations")
    with gr.Row():
        with gr.Column():
            transformer_path = gr.Textbox(
                label="Model Path",
                value=os.path.join(MODEL_DIR, "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"),
                info="Path to the model weights (transformer for Hunyuan, ckpt_path for Wan)"
            )
            vae_path = gr.Textbox(
                label="VAE Path (Hunyuan only)",
                value=os.path.join(MODEL_DIR, "hunyuan_video_vae_fp32.safetensors"),
                info="Path to the VAE model file (only used for Hunyuan model)."
            )
            llm_path = gr.Textbox(
                label="LLM Path (Hunyuan only)",
                value=os.path.join(MODEL_DIR, "llava-llama-3-8b-text-encoder-tokenizer"),
                info="Path to the LLM's text tokenizer and encoder (only used for Hunyuan model)."
            )
            clip_path = gr.Textbox(
                label="CLIP Path (Hunyuan only)",
                value=os.path.join(MODEL_DIR, "clip-vit-large-patch14"),
                info="Path to the CLIP model directory (only used for Hunyuan model)."
            )
    
    # Update model path placeholders based on model type
    def update_model_paths(model_type_value):
        if model_type_value == "hunyuan-video":
            return (
                gr.update(
                    label="Transformer Path",
                    value=os.path.join(MODEL_DIR, "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"),
                    info="Path to the transformer model weights for Hunyuan Video.",
                    visible=True
                ),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )
        else:  # wan model
            return (
                gr.update(
                    label="Checkpoint Path",
                    value="/data2/imagegen_models/Wan2.1-T2V-1.3B",
                    info="Path to the Wan2.1 model checkpoint.",
                    visible=True
                ),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    model_type.change(
        fn=update_model_paths,
        inputs=[model_type],
        outputs=[transformer_path, vae_path, llm_path, clip_path]
    )
            
    gr.Markdown("### Step 2: Training\nConfigure your training parameters and start or stop the training process.")
    with gr.Column():
        gr.Markdown("#### Training Parameters")
        with gr.Row():
            epochs = gr.Number(
                label="Epochs",
                value=1000,
                info="Total number of training epochs, Total Steps = ((Size of Dataset * Dataset Num Repeats) / (Batch Size * Gradient Accumulation Steps)) * Epochs"
            )
            batch_size = gr.Number(
                label="Batch Size",
                value=1,
                info="Batch size per GPU"
            )
            lr = gr.Number(
                label="Learning Rate",
                value=2e-5,
                step=0.0001,
                info="Optimizer learning rate"
            )
            save_every = gr.Number(
                label="Save Every N Epochs",
                value=2,
                info="Frequency to save checkpoints"
            )
        with gr.Row():
            eval_every = gr.Number(
                label="Evaluate Every N Epochs",
                value=1,
                info="Frequency to perform evaluations"
            )
            rank = gr.Number(
                label="LoRA Rank",
                value=32,
                info="LoRA adapter rank"
            )
            dtype = gr.Dropdown(
                label="LoRA Dtype",
                choices=['float32', 'float16', 'bfloat16', 'float8'],
                value="bfloat16",
            )
            gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                value=4,
                info="Micro-batch accumulation steps"
            )
        
        # Dataset configuration fields
        gr.Markdown("#### Dataset Configuration")
        with gr.Row():
            enable_ar_bucket = gr.Checkbox(
                label="Enable AR Bucket",
                value=True,
                info="Enable aspect ratio bucketing (Min and max aspect ratios, given as width/height ratio.)"
            )
            
        with gr.Row():
            min_ar = gr.Number(
                label="Minimum Aspect Ratio",
                value=0.5,
                step=0.1,
                info="Minimum aspect ratio for AR buckets"
            )
            max_ar = gr.Number(
                label="Maximum Aspect Ratio",
                value=2.0,
                step=0.1,
                info="Maximum aspect ratio for AR buckets"
            )
            num_ar_buckets = gr.Number(
                label="Number of AR Buckets",
                value=7,
                step=1,
                info="Number of aspect ratio buckets (Total number of aspect ratio buckets, evenly spaced (in log space) between min_ar and max_ar)"
            )
            
        with gr.Row():
            ar_buckets = gr.Textbox(
                label="AR Buckets (optional)",
                value="",
                info="Can manually specify AR Buckets instead of using the range-style config above, width/height ratio, or (width, height) pair. Example: [[512, 512], [448, 576]] or [1.0, 1.5]"
            )
            frame_buckets = gr.Textbox(
                label="Frame Buckets",
                value="[1, 33]",
                info="Videos will be assigned to the first frame bucket that the video is greater than or equal to in length. Example: [1, 33], 1 for images, if you have > 24GB or multiple GPUs: [1, 33, 65, 97]"
            )
        
        with gr.Row():
            num_repeats = gr.Number(
                label="Dataset Num Repeats",
                value=10,
                info="Number of times to duplicate the dataset"
            )
            resolutions_input = gr.Textbox(
                label="Resolutions",
                value="[512]",
                info="Resolutions to train on, given as a list. Example: [512] or [512, 768, 1024] or [[512, 512], [1280, 720]], defining only one side it will be a square, [512] = 512x512"
            )
                
        gr.Markdown("#### Optimizer Parameters")
        with gr.Row():
            optimizer_type = gr.Dropdown(
                label="Optimizer Type",
                choices=['adamw', 'adamw8bit', 'adamw_optimi', 'stableadamw', 'sgd', 'adamw8bitKahan', 'offload'],
                value="adamw_optimi",
            )
            betas = gr.Textbox(
                label="Betas",
                value="[0.9, 0.99]",
                info="Betas for the optimizer"
            )
        with gr.Row():
            weight_decay = gr.Number(
                label="Weight Decay",
                value=0.01,
                step=0.0001,
                info="Weight decay for regularization"
            )
            eps = gr.Number(
                label="Epsilon",
                value=1e-8,
                step=0.0000001,
                info="Epsilon for the optimizer"
            )
        
        # Additional training parameters
        gr.Markdown("#### Additional Training Parameters")
        with gr.Row():
            gradient_clipping = gr.Number(
                label="Gradient Clipping",
                value=1.0,
                step=0.1,
                info="Value for gradient clipping"
            )
            warmup_steps = gr.Number(
                label="Warmup Steps",
                value=100,
                step=10,
                info="Number of warmup steps"
            )
        with gr.Row():
            eval_before_first_step = gr.Checkbox(
                label="Evaluate Before First Step",
                value=True,
                info="Perform evaluation before the first training step"
            )
            eval_micro_batch_size_per_gpu = gr.Number(
                label="Eval Micro Batch Size Per GPU",
                value=1,
                info="Batch size for evaluation per GPU"
            )
            eval_gradient_accumulation_steps = gr.Number(
                label="Eval Gradient Accumulation Steps",
                value=1,
                info="Gradient accumulation steps for evaluation"
            )
            checkpoint_every_n_minutes = gr.Number(
                label="Checkpoint Every N Minutes",
                value=120,
                info="""Frequency to create checkpoints (in minutes), Used to restore training. 
                Note: Be careful with the time set here as the saved checkpoints take up a lot of disk space."""
            )
        with gr.Row():
            activation_checkpointing = gr.Checkbox(
                label="Activation Checkpointing",
                value=True,
                info="Enable activation checkpointing to save memory"
            )
            partition_method = gr.Textbox(
                label="Partition Method",
                value="parameters",
                info="Method for partitioning (e.g., parameters)"
            )
            save_dtype = gr.Dropdown(
                label="Save Dtype",
                choices=['bfloat16', 'float16', 'float32'],
                value="bfloat16",
                info="Data type to save model checkpoints"
            )
        with gr.Row():
            caching_batch_size = gr.Number(
                label="Caching Batch Size",
                value=1,
                info="Batch size for caching"
            )
            steps_per_print = gr.Number(
                label="Steps Per Print",
                value=1,
                info="Frequency to print logs to console"
            )
            video_clip_mode = gr.Textbox(
                label="Video Clip Mode",
                value="single_middle",
                info="""single_beginning: one clip starting at the beginning of the video,
                       single_middle: default, one clip from the middle of the video (cutting off the start and end equally),
                       multiple_overlapping: extract the minimum number of clips to cover the full range of the video. They might overlap some."""
            )
        
        gr.Markdown("#### Monitoring Settings")
        with gr.Row():
            enable_wandb = gr.Checkbox(
                label="Enable Wandb",
                value=False,
                info="Enable Wandb monitoring"
            )
            
            wandb_run_name = gr.Textbox(
                label="Wandb Run Name",
                info="Name of the wandb run",
                visible=False
            )
            
            wandb_tracker_name = gr.Textbox(
                label="Wandb Tracker Name",
                info="Name of the wandb tracker",
                visible=False
            )
            
            wandb_api_key = gr.Textbox(
                label="Wandb API Key",
                info="Wandb API Key (https://wandb.ai/authorize)",
                visible=False
            )
            
            def toggle_enable_wandb(checked):
                return gr.update(visible=checked), gr.update(visible=checked), gr.update(visible=checked)

            enable_wandb.change(
                toggle_enable_wandb,
                inputs=enable_wandb,
                outputs=[wandb_run_name, wandb_tracker_name, wandb_api_key]
            )     
    
        with gr.Row():
            with gr.Column(scale=1):
                resume_from_checkpoint = gr.Checkbox(label="Resume from last checkpoint", info="If this is your first training, do not check this box, because the output folder will not have a checkpoint (global_step....) and will cause an error")
                
                only_double_blocks = gr.Checkbox(label="Train only double blocks (Experimental)", info="This option will be used to train only double blocks, some people report that training only double blocks can reduce the amount of motion blur and improve the final quality of the video.")
                
                train_button = gr.Button("Start Training", visible=True)
                stop_button = gr.Button("Stop Training", visible=False)
                
                # Add fields for displaying current step and epoch
                with gr.Row():
                    total_steps = gr.State(0)
                    steps_per_epoch = gr.State(0)
                    current_epoch_display = gr.Textbox(label="Epoch Progress", interactive=False, value="Epoch: N/A")
                    current_step_display = gr.Textbox(label="Step Progress", interactive=False, value="Step: N/A")
                    
                with gr.Row():
                    force_save_model_button = gr.Button("Force Save Model", visible=False)
                    force_save_checkpoint_button = gr.Button("Force Save Checkpoint", visible=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        output = gr.Textbox(
                            label="Output Logs",
                            lines=20,
                            interactive=False,
                            elem_id="log_box"
                        )
                
                def force_save_model(output_dir_path):
                    force_save(output_dir_path, "save_model")
                    
                def force_save_checkpoint(output_dir_path):
                    force_save(output_dir_path, "save")
                    
                force_save_model_button.click(
                    fn=force_save_model,
                    inputs=[output_dir]
                )
                
                force_save_checkpoint_button.click(
                    fn=force_save_checkpoint,
                    inputs=[output_dir]
                )
                        
    hidden_config = gr.JSON(label="Hidden Configuration", visible=False)
    
     # Adding Download Section
    gr.Markdown("### Download Files")
    
    with gr.Row():
        explorer = gr.FileExplorer(root_dir="/workspace", interactive=True, label="File Explorer", file_count="single", elem_id="file_explorer")     
        with gr.Column():   
            gr.Markdown("### Select a single file from the file explorer for download.")
            download_status = gr.Textbox(label="Single File Download Status", interactive=False)
            download_file = gr.File(label="Download Single File", interactive=False)
        

            # Event: When a file is selected in the explorer
        explorer.change(
            fn=get_selected_file,
            inputs=[explorer],
            outputs=[download_file, download_status],
        )
            
    with gr.Row():
        with gr.Column():
            download_options = gr.CheckboxGroup(["Outputs", "Dataset", "Configs"], label="Bulk Download Options"),
            download_button = gr.Button("Download ZIP", visible=True)
        download_zip = gr.File(label="Download ZIP", visible=True)
        download_status = gr.Textbox(label="Bulk Download Status", interactive=False, visible=True)

    
    def handle_train_click(
        dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
        transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
        gradient_accumulation_steps, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, ar_buckets, gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print, video_clip_mode, resume_from_checkpoint, only_double_blocks, enable_wandb, wandb_run_name, wandb_tracker_name, wandb_api_key, model_type
    ):
        
        with process_lock:
            if process_dict:
                return "A training process is already running. Please stop it before starting a new one.", training_process_pid, gr.update(interactive=False)
            
        message, pid = train_model(
            dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
            gradient_accumulation_steps, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, ar_buckets, gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print, video_clip_mode, resume_from_checkpoint, only_double_blocks, enable_wandb, wandb_run_name, wandb_tracker_name, wandb_api_key, model_type
        )
        
        if pid:
            # Disable the training button while training is active
            return message, pid, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return message, pid, gr.update(visible=True), gr.update(visible=False),  gr.update(visible=False),  gr.update(visible=False)

    def handle_stop_click(pid):
        message = stop_training(pid)
        return message, gr.update(visible=True), gr.update(visible=False)

    def refresh_logs(
        log_box, 
        pid, 
        current_epoch_display, 
        current_step_display, 
        total_steps, 
        steps_per_epoch, 
        last_step, 
        last_epoch
    ):
        new_logs = ""
        updated_epoch = last_epoch
        updated_step = last_step
        total_steps_value = total_steps
        steps_per_epoch_value = steps_per_epoch

        while not log_queue.empty():
            log_line = log_queue.get()
            new_logs += log_line

            # Extract total steps
            if "Total steps:" in log_line:
                try:
                    total_steps_value = int(log_line.split("Total steps:")[1].strip())
                except Exception as e:
                    print(f"Error parsing total steps: {e}")

            # Extract steps per epoch
            if "Steps per epoch:" in log_line:
                try:
                    steps_per_epoch_value = int(log_line.split("Steps per epoch:")[1].strip())
                except Exception as e:
                    print(f"Error parsing steps per epoch: {e}")

            # Extract step information
            if "step=" in log_line:
                try:
                    updated_step = int(log_line.split("step=")[1].split(",")[0].strip())
                except Exception as e:
                    print(f"Error parsing step info: {e}")

            # Extract epoch information
            if "Started new epoch:" in log_line:
                try:
                    updated_epoch = int(log_line.split("Started new epoch:")[1].strip())
                except Exception as e:
                    print(f"Error parsing epoch info: {e}")

        # Calculate progress
        total_epochs = total_steps_value // steps_per_epoch_value if steps_per_epoch_value else 0
        if updated_epoch is None:
            updated_epoch = 1  # Default to epoch 1 if no epoch data is available

        epoch_progress = f"Epoch: {updated_epoch} / {total_epochs}"
        step_percentage = (updated_step / total_steps_value * 100) if total_steps_value else 0
        step_progress = f"Step: {updated_step} / {total_steps_value} ({step_percentage:.1f}%)"

        # Return updated values
        return (
            log_box + new_logs,
            epoch_progress,
            step_progress,
            total_steps_value,
            steps_per_epoch_value,
            updated_step,
            updated_epoch,
        )


    # Persistent states for step and epoch tracking
    last_step = gr.State(0)
    last_epoch = gr.State(1)

    # Timer to refresh logs
    log_timer = gr.Timer(0.5, active=False)

    # Connect refresh_logs to update the log box and progress displays
    log_timer.tick(
        fn=refresh_logs,
        inputs=[
            output,                  # Log box
            training_process_pid,    # Process ID
            current_epoch_display,   # Current epoch display
            current_step_display,    # Current step display
            total_steps,             # Total steps state
            steps_per_epoch,         # Steps per epoch state
            last_step,               # Last step state
            last_epoch,              # Last epoch state
        ],
        outputs=[
            output,                  # Updated log box
            current_epoch_display,   # Updated epoch display
            current_step_display,    # Updated step display
            total_steps,             # Updated total steps state
            steps_per_epoch,         # Updated steps per epoch state
            last_step,               # Updated last step state
            last_epoch,              # Updated last epoch state
        ]
    )

    def activate_timer():
        return gr.update(active=True)
                    
    train_click = train_button.click(
        fn=handle_train_click,
        inputs=[
            dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
            gradient_accumulation_steps, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar,
            num_ar_buckets, frame_buckets, ar_buckets, gradient_clipping, warmup_steps, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes,
            activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
            video_clip_mode, resume_from_checkpoint, only_double_blocks, enable_wandb, wandb_run_name, 
            wandb_tracker_name, wandb_api_key, model_type
        ],
        outputs=[output, training_process_pid, train_button, stop_button, force_save_model_button, force_save_checkpoint_button],
        api_name=None
    ).then(
        fn=lambda: gr.update(active=True),  # Activate the Timer
        inputs=None,
        outputs=log_timer
    )
    
    def deactivate_timer():
        return gr.update(active=False)
    
    stop_click = stop_button.click(
        fn=handle_stop_click,
        inputs=[training_process_pid],
        outputs=[output,train_button, stop_button],
        api_name=None
    ).then(
        fn=lambda: gr.update(active=False),  # Deactivate the Timer
        inputs=None,
        outputs=log_timer
    )
    
    
    # Handle Download Button Click
    download_button.click(
        fn=handle_download,
        inputs=[dataset_path, download_options[0]],
        outputs=[download_zip, download_status],
        queue=True
    )
    
    with gr.Row():
        download_zip
    
    # Ensure that the "Download ZIP" button is only visible when a dataset is selected or created
    download_button.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=download_zip
    )
    
    # Handle selecting an existing dataset
    existing_datasets.change(
        fn=handle_select_existing,
        inputs=existing_datasets,
        outputs=[
            dataset_path, 
            config_dir, 
            output_dir, 
            upload_status, 
            gallery,
            # download_button,
            download_status,
            hidden_config 
        ]
    ).then(
        fn=lambda config_vals: update_ui_with_config(config_vals),
        inputs=hidden_config,  # Receives configuration values
        outputs=[
            epochs, batch_size, lr, save_every, eval_every, rank, only_double_blocks, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type,
            betas, weight_decay, eps, gradient_accumulation_steps, num_repeats,
            resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, ar_buckets,
            frame_buckets, gradient_clipping, warmup_steps, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
            checkpoint_every_n_minutes, activation_checkpointing, partition_method,
            save_dtype, caching_batch_size, steps_per_print, video_clip_mode, enable_wandb,
            wandb_run_name, wandb_tracker_name, wandb_api_key, model_type
        ]
    )
    
    # Handle dataset creation
    create_dataset_button.click(
        fn=handle_start_dataset,
        inputs=[dataset_name_input],
        outputs=[upload_status, current_dataset_state, create_dataset_button, upload_files, dataset_path]
    )
    
    # Handle dataset option change
    dataset_option.change(
        fn=toggle_dataset_option,
        inputs=[dataset_option],
        outputs=[
            create_new_container,
            select_existing_container,
            existing_datasets,
            dataset_name_input,
            upload_status,
            dataset_path,
            create_dataset_button,
            upload_files
        ]
    )
    
def parse_args():
    parser = argparse.ArgumentParser(description="Gradio Interface for LoRA Training on Video Generation Models")

    parser.add_argument("--local", action="store_true", help="Run the interface locally")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.local:
        MODEL_DIR = os.path.join(os.getcwd(), "models")
        BASE_DATASET_DIR = os.path.join(os.getcwd(), "datasets")
        OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
        CONFIG_DIR = os.path.join(os.getcwd(), "configs")
        
    # Create directories if they don't exist
    for dir_path in [MODEL_DIR, BASE_DATASET_DIR, OUTPUT_DIR, CONFIG_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace", ".", os.getcwd()])