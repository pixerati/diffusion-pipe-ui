import queue
import signal
import subprocess
import threading
import gradio as gr
import os
from datetime import datetime
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

DEVELOPMENT = False

# Working directories
MODEL_DIR = os.path.join(os.getcwd(), "models") if DEVELOPMENT else "/workspace/models"
BASE_DATASET_DIR = os.path.join(os.getcwd(), "datasets") if DEVELOPMENT else "/workspace/datasets"
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs") if DEVELOPMENT else "/workspace/outputs"
CONFIG_DIR = os.path.join(os.getcwd(), "configs") if DEVELOPMENT else "/workspace/configs"

# Maximum number of media to display in the gallery
MAX_MEDIA = 50

# Determine if running on Runpod by checking the environment variable
IS_RUNPOD = os.getenv("IS_RUNPOD", "false").lower() == "true"

CONDA_DIR = os.getenv("CONDA_DIR", "/opt/conda")  # Directory where Conda is installed in the Docker container

# Maximum upload size in MB (Gradio expects max_file_size in MB)
MAX_UPLOAD_SIZE_MB = 500 if IS_RUNPOD else None  # 500MB or no limit

# Create directories if they don't exist
for dir_path in [MODEL_DIR, BASE_DATASET_DIR, OUTPUT_DIR, CONFIG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

process_dict = {}
process_lock = threading.Lock()

log_queue = queue.Queue()

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
                        frame_buckets: list) -> str:
    """Create and save the dataset configuration in TOML format."""
    dataset_config = {
        "resolutions": resolutions,
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": frame_buckets,
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
    transformer_path: str,
    vae_path: str,
    llm_path: str,
    clip_path: str,
    dtype: str = "float16",
    
    # Adapter parameters
    rank: int = 8,
    
    # Optimizer parameters
    optimizer_type: str = "adamw_optimi",
    lr: float = 1e-4,
    betas: str = "(0.9, 0.999)",
    weight_decay: float = 0.01,
    eps: float = 1e-8
):
    """
    Creates a training configuration dictionary from individual parameters.
    """
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
        # Model configuration with fixed type and sampling method
        "model": {
            "type": "hunyuan-video",
            "transformer_path": transformer_path,
            "vae_path": vae_path,
            "llm_path": llm_path,
            "clip_path": clip_path,
            "dtype": dtype,
            "transformer_dtype": "float8",
            "timestep_sample_method": "logit_normal"
        },
        # Adapter configuration with fixed type
        "adapter": {
            "type": "lora",
            "rank": rank,
            "dtype": dtype
        },
        # Optimizer configuration with evaluated betas
        "optimizer": {
            "type": optimizer_type,
            "lr": lr,
            "betas": eval(betas),
            "weight_decay": weight_decay,
            "eps": eps
        }
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
    dtype = config.get("adapter", {}).get("dtype", "bfloat16")
    transformer_path = config.get("model", {}).get("transformer_path", "")
    vae_path = config.get("model", {}).get("vae_path", "")
    llm_path = config.get("model", {}).get("llm_path", "")
    clip_path = config.get("model", {}).get("clip_path", "")
    optimizer_type = config.get("optimizer", {}).get("type", "adamw_optimi")
    betas = config.get("optimizer", {}).get("betas", [0.9, 0.99])
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.01)
    eps = config.get("optimizer", {}).get("eps", 1e-8)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    num_repeats = config.get("dataset", {}).get("num_repeats", 10)
    resolutions = config.get("dataset", {}).get("resolutions", [512])
    enable_ar_bucket = config.get("dataset", {}).get("enable_ar_bucket", True)
    min_ar = config.get("dataset", {}).get("min_ar", 0.5)
    max_ar = config.get("dataset", {}).get("max_ar", 2.0)
    num_ar_buckets = config.get("dataset", {}).get("num_ar_buckets", 7)
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
    
    return {
        "epochs": training_params,
        "batch_size": batch_size,
        "lr": lr,
        "save_every": save_every,
        "eval_every": eval_every,
        "rank": rank,
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
        "video_clip_mode": video_clip_mode
    }

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
            gr.update(visible=False, value=None),    # Hide Upload Files
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
            gr.update(visible=False),     # Hide Create Dataset Button
            gr.update(visible=False, value=None),    # Hide Upload Files
        )
                
def train_model(dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
                gradient_accumulation_steps, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
                gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
                checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
                video_clip_mode,
                # num_gpus,  # Added parameter num_gpus
                ):
    try:
        # Validate inputs
        if not dataset_path or not os.path.exists(dataset_path) or dataset_path == BASE_DATASET_DIR:
            return "Error: Please provide a valid dataset path"
        
        os.makedirs(config_dir, exist_ok=True)

        if not config_dir or not os.path.exists(config_dir) or config_dir == CONFIG_DIR:
            return "Error: Please provide a valid config path"

        os.makedirs(output_dir, exist_ok=True)
        
        if not output_dir or not os.path.exists(output_dir) or output_dir == OUTPUT_DIR:
            return "Error: Please provide a valid output path"
        
        try:
            resolutions_list = json.loads(resolutions)
            if not isinstance(resolutions_list, list) or not all(isinstance(r, int) for r in resolutions_list):
                return "Error: Resolutions must be a list of integers. Example: [512] or [512, 768, 1024]"
        except Exception as e:
            return f"Error parsing resolutions: {str(e)}"
        
        try:
            frame_buckets_list = json.loads(frame_buckets)
            if not isinstance(frame_buckets_list, list) or not all(isinstance(b, int) for b in frame_buckets_list):
                return "Error: Frame buckets must be a list of integers. Example: [1, 33, 65]"
        except Exception as e:
            return f"Error parsing frame buckets: {str(e)}"

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
            frame_buckets=frame_buckets_list
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
            video_clip_mode=video_clip_mode
        )

        # Create args to pass to the train function
        # args = argparse.Namespace(
        #     local_rank=-1,
        #     resume_from_checkpoint=False,
        #     regenerate_cache=False,
        #     cache_only=False
        # )
        
        conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
        conda_env_name = "pyenv"
        # conda_activate_path = os.path.join(CONDA_DIR, "etc", "profile.d", "conda.sh")
        # conda_env_name = "pyenv"
        
        if not os.path.isfile(conda_activate_path):
            return "Error: Conda activation script not found"
        
        cmd = (
            f"bash -c 'source {conda_activate_path} && "
            f"conda activate {conda_env_name} && "
            f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 "
            f"train.py --deepspeed --config {training_config_path}'"          
        )
            
        proc = subprocess.Popen(
            cmd,
            shell=True,  # Required for complex shell commands
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            #text=True,
            # start_new_session=True, 
            preexec_fn=os.setsid,
            universal_newlines=False  # To handle bytes
        )
        
        with process_lock:
            process_dict[proc.pid] = proc  
        
        thread = threading.Thread(target=read_subprocess_output, args=(proc, log_queue))
        thread.start()
        
        pid = proc.pid
        
        return "Training started! Logs will appear below. \n", pid

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
            return current_dataset, "Please provide a dataset name."
        # Ensure the dataset name does not contain invalid characters
        dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        dataset_dir = os.path.join(BASE_DATASET_DIR, dataset_name)
        if os.path.exists(dataset_dir):
            return current_dataset, f"Dataset '{dataset_name}' already exists. Please choose a different name."
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir, f"Started new dataset: {dataset_dir}"

    if not current_dataset:
        return current_dataset, "Please start a new dataset before uploading files."

    if not files:
        return current_dataset, "No files uploaded."

    # Calculate the total size of the current dataset
    total_size = 0
    for root, dirs, files_in_dir in os.walk(current_dataset):
        for f in files_in_dir:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)

    # Calculate the size of the new files
    new_files_size = 0
    for file in files:
        if IS_RUNPOD:
            new_files_size += os.path.getsize(file.name)

    # Check if adding these files would exceed the limit
    if IS_RUNPOD and (total_size + new_files_size) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return current_dataset, f"Upload would exceed the {MAX_UPLOAD_SIZE_MB}MB limit on Runpod. Please upload smaller files or finalize the dataset."

    uploaded_files = []

    for file in files:
        file_path = file.name
        filename = os.path.basename(file_path)
        dest_path = os.path.join(current_dataset, filename)

        if zipfile.is_zipfile(file_path):
            # If the file is a ZIP, extract its contents
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(current_dataset)
                uploaded_files.append(f"{filename} (extracted)")
            except zipfile.BadZipFile:
                uploaded_files.append(f"{filename} (invalid ZIP)")
                continue
        else:
            # Check if the file is a supported format
            if filename.lower().endswith(('.mp4', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.txt')):
                shutil.copy(file_path, dest_path)
                uploaded_files.append(filename)
            else:
                uploaded_files.append(f"{filename} (unsupported format)")
                continue

    return current_dataset, f"Uploaded files: {', '.join(uploaded_files)}"

def update_ui_with_config(config_values):
    """
    Updates Gradio interface components with configuration values.

    Args:
        config_values (dict): Dictionary containing dataset and training configurations.

    Returns:
        tuple: Updated values for the interface components.
    """
    # Define default values for each field
    defaults = {
        "epochs": 1000,
        "batch_size": 1,
        "lr": 2e-5,
        "save_every": 2,
        "eval_every": 1,
        "rank": 32,
        "dtype": "bfloat16",
        "transformer_path": "",
        "vae_path": "",
        "llm_path": "",
        "clip_path": "",
        "optimizer_type": "adamw_optimi",
        "betas": json.dumps([0.9, 0.99]),
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_accumulation_steps": 4,
        "num_repeats": 10,
        "resolutions_input": json.dumps([512]),
        "enable_ar_bucket": True,
        "min_ar": 0.5,
        "max_ar": 2.0,
        "num_ar_buckets": 7,
        "frame_buckets": json.dumps([1, 33, 65]),
        "gradient_clipping": 1.0,
        "warmup_steps": 100,
        "eval_before_first_step": True,
        "eval_micro_batch_size_per_gpu": 1,
        "eval_gradient_accumulation_steps": 1,
        "checkpoint_every_n_minutes": 120,
        "activation_checkpointing": True,
        "partition_method": "parameters",
        "save_dtype": "bfloat16",
        "caching_batch_size": 1,
        "steps_per_print": 1,
        "video_clip_mode": "single_middle"
    }

    # Helper function to get values with defaults
    def get_value(key):
        return config_values.get(key, defaults.get(key))

    # Extract values with error handling
    try:
        epochs = get_value("epochs")
        batch_size = get_value("batch_size")
        lr = get_value("lr")
        save_every = get_value("save_every")
        eval_every = get_value("eval_every")
        rank = get_value("rank")
        dtype = get_value("dtype")
        transformer_path = get_value("transformer_path")
        vae_path = get_value("vae_path")
        llm_path = get_value("llm_path")
        clip_path = get_value("clip_path")
        optimizer_type = get_value("optimizer_type")
        betas = get_value("betas")
        weight_decay = get_value("weight_decay")
        eps = get_value("eps")
        gradient_accumulation_steps = get_value("gradient_accumulation_steps")
        num_repeats = get_value("num_repeats")
        resolutions_input = get_value("resolutions_input")
        enable_ar_bucket = get_value("enable_ar_bucket")
        min_ar = get_value("min_ar")
        max_ar = get_value("max_ar")
        num_ar_buckets = get_value("num_ar_buckets")
        frame_buckets = get_value("frame_buckets")
        gradient_clipping = get_value("gradient_clipping")
        warmup_steps = get_value("warmup_steps")
        eval_before_first_step = get_value("eval_before_first_step")
        eval_micro_batch_size_per_gpu = get_value("eval_micro_batch_size_per_gpu")
        eval_gradient_accumulation_steps = get_value("eval_gradient_accumulation_steps")
        checkpoint_every_n_minutes = get_value("checkpoint_every_n_minutes")
        activation_checkpointing = get_value("activation_checkpointing")
        partition_method = get_value("partition_method")
        save_dtype = get_value("save_dtype")
        caching_batch_size = get_value("caching_batch_size")
        steps_per_print = get_value("steps_per_print")
        video_clip_mode = get_value("video_clip_mode")
    except Exception as e:
        print(f"Error extracting configurations: {str(e)}")
        # Return default values in case of an error
        epochs = defaults["epochs"]
        batch_size = defaults["batch_size"]
        lr = defaults["lr"]
        save_every = defaults["save_every"]
        eval_every = defaults["eval_every"]
        rank = defaults["rank"]
        dtype = defaults["dtype"]
        transformer_path = defaults["transformer_path"]
        vae_path = defaults["vae_path"]
        llm_path = defaults["llm_path"]
        clip_path = defaults["clip_path"]
        optimizer_type = defaults["optimizer_type"]
        betas = defaults["betas"]
        weight_decay = defaults["weight_decay"]
        eps = defaults["eps"]
        gradient_accumulation_steps = defaults["gradient_accumulation_steps"]
        num_repeats = defaults["num_repeats"]
        resolutions_input = defaults["resolutions_input"]
        enable_ar_bucket = defaults["enable_ar_bucket"]
        min_ar = defaults["min_ar"]
        max_ar = defaults["max_ar"]
        num_ar_buckets = defaults["num_ar_buckets"]
        frame_buckets = defaults["frame_buckets"]
        gradient_clipping = defaults["gradient_clipping"]
        warmup_steps = defaults["warmup_steps"]
        eval_before_first_step = defaults["eval_before_first_step"]
        eval_micro_batch_size_per_gpu = defaults["eval_micro_batch_size_per_gpu"]
        eval_gradient_accumulation_steps = defaults["eval_gradient_accumulation_steps"]
        checkpoint_every_n_minutes = defaults["checkpoint_every_n_minutes"]
        activation_checkpointing = defaults["activation_checkpointing"]
        partition_method = defaults["partition_method"]
        save_dtype = defaults["save_dtype"]
        caching_batch_size = defaults["caching_batch_size"]
        steps_per_print = defaults["steps_per_print"]
        video_clip_mode = defaults["video_clip_mode"]
    print(num_repeats)
    return (
        epochs,
        batch_size,
        lr,
        save_every,
        eval_every,
        rank,
        dtype,
        transformer_path,
        vae_path,
        llm_path,
        clip_path,
        optimizer_type,
        betas,
        weight_decay,
        eps,
        gradient_accumulation_steps,
        num_repeats,
        resolutions_input,
        enable_ar_bucket,
        min_ar,
        max_ar,
        num_ar_buckets,
        frame_buckets,
        gradient_clipping,
        warmup_steps,
        eval_before_first_step,
        eval_micro_batch_size_per_gpu,
        eval_gradient_accumulation_steps,
        checkpoint_every_n_minutes,
        activation_checkpointing,
        partition_method,
        save_dtype,
        caching_batch_size,
        steps_per_print,
        video_clip_mode
    )

def show_media(dataset_dir):
    """Display uploaded images and .mp4 videos in a single gallery."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        # Return an empty list if the dataset_dir is invalid
        return []

    # List of image and .mp4 video files
    media_files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.mp4'))
    ]

    # Get absolute paths of the files
    media_paths = [os.path.abspath(os.path.join(dataset_dir, f)) for f in media_files[:MAX_MEDIA]]

    # Check if the files exist
    existing_media = [f for f in media_paths if os.path.exists(f)]

    return existing_media

theme = gr.themes.Monochrome(
    primary_hue="gray",
    secondary_hue="gray",
    neutral_hue="gray",
    text_size=gr.themes.Size(
        lg="18px", 
        md="15px", 
        sm="13px", 
        xl="22px", 
        xs="12px", 
        xxl="24px", 
        xxs="9px"
    ),
    font=[
        gr.themes.GoogleFont("Source Sans Pro"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif"
    ]
)

# Gradio Interface
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# LoRA Training Interface for Hunyuan Video")
    
    gr.Markdown("### Step 1: Dataset Management\nChoose to create a new dataset or select an existing one.")
    
    with gr.Row():
        dataset_option = gr.Radio(
            choices=["Create New Dataset", "Select Existing Dataset"],
            value="Create New Dataset",
            label="Dataset Option"
        )
    
    def handle_start_dataset(dataset_name):
        if not dataset_name.strip():
            return (
                gr.update(value="Please provide a dataset name."), 
                gr.update(value=None), 
                gr.update(visible=True),   # Keep button visible
                gr.update(visible=False),
                gr.update(value="")        # Clear dataset_path_display
            )
        dataset_path, message = upload_dataset([], None, "start", dataset_name=dataset_name)
        if "already exists" in message:
            return (
                gr.update(value=message), 
                gr.update(value=None), 
                gr.update(visible=True),   # Keep button visible
                gr.update(visible=False),
                gr.update(value="")        # Clear dataset_path_display
            )
        return (
            gr.update(value=message), 
            dataset_path, 
            gr.update(visible=False), 
            gr.update(visible=True),
            gr.update(value=dataset_path)  # Update dataset_path_display
        )
    

    with gr.Row(visible=True, elem_id="create_new_dataset_container") as create_new_container:
            with gr.Column():
                with gr.Row():
                    dataset_name_input = gr.Textbox(
                        label="Dataset Name",
                        placeholder="Enter your dataset name",
                        interactive=True
                    )
                create_dataset_button = gr.Button("Create Dataset", interactive=False)  # Initially disabled
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                upload_files = gr.File(
                    label="Upload Images (.jpg, .png, .gif, .bmp, .webp), Videos (.mp4), Captions (.txt) or a ZIP archive",
                    file_types=[".jpg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".txt", ".zip"],
                    file_count="multiple",
                    type="filepath", 
                    interactive=True,
                    visible=False
                )
                
    # Function to enable/disable the "Start New Dataset" button based on input
    def toggle_start_button(name):
        if name.strip():
            return gr.update(interactive=True)
        else:
            return gr.update(interactive=False)
    
    current_dataset_state = gr.State(None)
    training_process_pid = gr.State(None)
    
    dataset_name_input.change(
        fn=toggle_start_button, 
        inputs=dataset_name_input, 
        outputs=create_dataset_button
    )
    
    def handle_upload(files, current_dataset):
        updated_dataset, message = upload_dataset(files, current_dataset, "add")
        return updated_dataset, message, show_media(updated_dataset)
    
    
    # Container to select existing dataset
    with gr.Row(visible=False, elem_id="select_existing_dataset_container") as select_existing_container:
        with gr.Column():
            existing_datasets = gr.Dropdown(
                choices=[],  # Initially empty; will be updated dynamically
                label="Select Existing Dataset",
                interactive=True
            )
            # select_dataset_button = gr.Button("Select Dataset", interactive=False)  # Initially disabled
    
    # 2. Media Gallery
    gr.Markdown("### Dataset Preview")
    gallery = gr.Gallery(
        label="Dataset Preview",
        show_label=False,
        elem_id="gallery",
        columns=3,
        rows=2,
        object_fit="contain",
        height="auto",
        visible=True
    )
    
    # Upload files and update gallery
    upload_files.upload(
        fn=lambda files, current_dataset: handle_upload(files, current_dataset),
        inputs=[upload_files, current_dataset_state],
        outputs=[current_dataset_state, upload_status, gallery],
        queue=True
    )
    
    # Function to handle selecting an existing dataset and updating the gallery
    def handle_select_existing(selected_dataset):
        if selected_dataset:
            dataset_path = os.path.join(BASE_DATASET_DIR, selected_dataset)
            config, error = load_training_config(selected_dataset)
            if error:
                return (
                    "",  # Clear dataset path
                    "",  # Clear config and output paths
                    "",  # Clear parameter values
                    f"Error loading configuration: {error}",
                    []    # Clear gallery
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
                config_values  # Pass configuration values to be updated in the UI
            )
        return "", "", "", "No dataset selected.", [], {}
    
    with gr.Row():
        with gr.Column():
            dataset_path = gr.Textbox(
                label="Dataset Path",
                value=BASE_DATASET_DIR
            )
            config_dir = gr.Textbox(
                label="Config Path",
                value=CONFIG_DIR
            )
            output_dir = gr.Textbox(
                label="Output Path",
                value=OUTPUT_DIR
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
            
    gr.Markdown("#### Models Configurations")
    with gr.Row():
        with gr.Column():
            transformer_path = gr.Textbox(
                label="Transformer Path",
                value=os.path.join(MODEL_DIR, "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"),
                info="Path to the transformer model weights for Hunyuan Video."
            )
            vae_path = gr.Textbox(
                label="VAE Path",
                value=os.path.join(MODEL_DIR, "hunyuan_video_vae_fp32.safetensors"),
                info="Path to the VAE model file."
            )
            llm_path = gr.Textbox(
                label="LLM Path",
                value=os.path.join(MODEL_DIR, "llava-llama-3-8b-text-encoder-tokenizer"),
                info="Path to the LLM's text tokenizer and encoder."
            )
            clip_path = gr.Textbox(
                label="CLIP Path",
                value=os.path.join(MODEL_DIR, "clip-vit-large-patch14"),
                info="Path to the CLIP model directory."
            )
            
    gr.Markdown("### Step 2: Training\nConfigure your training parameters and start or stop the training process.")
    with gr.Column():
        gr.Markdown("#### Training Parameters")
        with gr.Row():
            epochs = gr.Number(
                label="Epochs",
                value=1000,
                info="Total number of training epochs"
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
                info="Enable aspect ratio bucketing"
            )
            min_ar = gr.Number(
                label="Minimum Aspect Ratio",
                value=0.5,
                step=0.1,
                info="Minimum aspect ratio for AR buckets"
            )
        with gr.Row():
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
                info="Number of aspect ratio buckets"
            )
            frame_buckets = gr.Textbox(
                label="Frame Buckets",
                value="[1, 33, 65]",
                info="Frame buckets as a JSON list. Example: [1, 33, 65]"
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
                info="Resolutions to train on, given as a list. Example: [512] or [512, 768, 1024]"
            )
                
        gr.Markdown("#### Optimizer Parameters")
        with gr.Row():
            optimizer_type = gr.Textbox(
                label="Optimizer Type",
                value="adamw_optimi",
                info="Type of optimizer"
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
                info="Frequency to create checkpoints (in minutes)"
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
                info="Frequency to print training steps"
            )
            video_clip_mode = gr.Textbox(
                label="Video Clip Mode",
                value="single_middle",
                info="Mode for video clipping (e.g., single_middle)"
            )
            
    
        with gr.Row():
            with gr.Column(scale=1):
                train_button = gr.Button("Start Training", visible=True)
                stop_button = gr.Button("Stop Training", visible=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        output = gr.Textbox(
                            label="Output Logs",
                            lines=20,
                            interactive=False,
                            elem_id="log_box"
                        )
    
    hidden_config = gr.JSON(label="Hidden Configuration", visible=False)
     
    # Event to update dataset path and gallery when selecting an existing one
    existing_datasets.change(
        fn=handle_select_existing,
        inputs=existing_datasets,
        outputs=[
            dataset_path, 
            config_dir, 
            output_dir, 
            upload_status, 
            gallery,
            hidden_config  # Add a state to store configuration values
        ]
    ).then(
        fn=lambda config_vals: update_ui_with_config(config_vals),
        inputs=hidden_config,  # Receives configuration values
        outputs=[
            epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type,
            betas, weight_decay, eps, gradient_accumulation_steps, num_repeats,
            resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets,
            frame_buckets, gradient_clipping, warmup_steps, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
            checkpoint_every_n_minutes, activation_checkpointing, partition_method,
            save_dtype, caching_batch_size, steps_per_print, video_clip_mode
        ]
    )
    
    def handle_train_click(
        dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
        transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
        gradient_accumulation_steps, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
        gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
        checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
        video_clip_mode,
        # num_gpus,  # Added parameter num_gpus
    ):
        
        with process_lock:
            if process_dict:
                return "A training process is already running. Please stop it before starting a new one.", training_process_pid, gr.update(interactive=False)
            
        message, pid = train_model(
            dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
            gradient_accumulation_steps, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
            gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
            checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
            video_clip_mode,
            # num_gpus,
        )
        
        if pid:
            # Disable the training button while training is active
            return message, pid, gr.update(visible=False), gr.update(visible=True)
        else:
            return message, pid, gr.update(visible=True), gr.update(visible=False)

    def handle_stop_click(pid):
        message = stop_training(pid)
        return message, gr.update(visible=True), gr.update(visible=False)

    def refresh_logs(log_box, pid):
        if pid is not None:
            return update_logs(log_box, pid)
        return log_box
    
    log_timer = gr.Timer(0.5, active=False) 
    
    log_timer.tick(
        fn=refresh_logs,
        inputs=[output, training_process_pid],
        outputs=output
    )
    
    def activate_timer():
        return gr.update(active=True)
    
    train_click = train_button.click(
        fn=handle_train_click,
        inputs=[
            dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
            gradient_accumulation_steps, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar,
            num_ar_buckets, frame_buckets, gradient_clipping, warmup_steps, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes,
            activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
            video_clip_mode
            # gr.Number(label="Number of GPUs", value=1, info="Number of GPUs to use"),
        ],
        outputs=[output, training_process_pid, train_button, stop_button],
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
    

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace", "."])
