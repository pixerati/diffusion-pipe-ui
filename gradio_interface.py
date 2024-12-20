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

# Importando funções necessárias do train.py
from train import train


DEVELOPMENT = False

# Diretórios de trabalho
MODEL_DIR = os.path.join(os.getcwd(), "models") if DEVELOPMENT else "/workspace/models"
BASE_DATASET_DIR = os.path.join(os.getcwd(), "datasets") if DEVELOPMENT else "/workspace/datasets"
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs") if DEVELOPMENT else "/workspace/outputs"
CONFIG_DIR = os.path.join(os.getcwd(), "configs") if DEVELOPMENT else "/workspace/configs"

# Maximum number of media to display in the gallery
MAX_MEDIA = 50

# Determine if running on Runpod by checking the environment variable
IS_RUNPOD = os.getenv("IS_RUNPOD", "false").lower() == "true"

# Maximum upload size in MB (Gradio expects max_file_size in MB)
MAX_UPLOAD_SIZE_MB = 500 if IS_RUNPOD else None  # 500MB or no limit

# Criar diretórios se não existirem
for dir_path in [MODEL_DIR, BASE_DATASET_DIR, OUTPUT_DIR, CONFIG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    
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
    optimizer_type: str = "adamw",
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

def toggle_dataset_option(option):
    if option == "Create New Dataset":
        # Mostrar container de criação e ocultar container de seleção
        return (
            gr.update(visible=True),    # Mostrar create_new_container
            gr.update(visible=False),   # Ocultar select_existing_container
            gr.update(choices=[], value=None),      # Limpar Dropdown de datasets existentes
            gr.update(value=""),        # Limpar Dataset Name
            gr.update(value=""),         # Limpar Upload Status
            gr.update(value=""),         # Limpar Dataset Path
            gr.update(visible=True),     # Ocultar Create Dataset Button
            gr.update(visible=False, value=None),    # Ocultar Upload Files
        )
    else:
        # Ocultar container de criação e mostrar container de seleção
        datasets = get_datasets()
        return (
            gr.update(visible=False),   # Ocultar create_new_container
            gr.update(visible=True),    # Mostrar select_existing_container
            gr.update(choices=datasets if datasets else [], value=None),  # Atualizar Dropdown
            gr.update(value=""),        # Limpar Dataset Name
            gr.update(value=""),         # Limpar Upload Status
            gr.update(value=""),         # Limpar Dataset Path
            gr.update(visible=False),     # Ocultar Create Dataset Button
            gr.update(visible=False, value=None),    # Ocultar Upload Files
        )
                
def train_model(dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
                gradient_accumulation_steps, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
                gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
                checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
                video_clip_mode):
    try:
        # Validar inputs
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

        # Criar configurações
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

        # Criar args para passar para a função train
        args = argparse.Namespace(
            local_rank=-1,
            resume_from_checkpoint=False,
            regenerate_cache=False,
            cache_only=False
        )

        # Chamar a função train do train.py
        run_dir = train(training_config_path, args)
        
        return f"Training completed successfully! Output directory: {run_dir}"

    except Exception as e:
        return f"Error during training: {str(e)}"
    
    
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


def show_media(dataset_dir):
    """Display uploaded images and .mp4 videos in a single gallery."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        # Retorna uma lista vazia se o dataset_dir for inválido
        return []

    # Lista de arquivos de imagem e vídeos .mp4
    media_files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.mp4'))
    ]

    # Obtenha os caminhos absolutos dos arquivos
    media_paths = [os.path.abspath(os.path.join(dataset_dir, f)) for f in media_files[:MAX_MEDIA]]

    # Verifique se os arquivos existem
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

# Interface Gradio
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
                gr.update(visible=True),   # Manter botão visível
                gr.update(visible=False),
                gr.update(value="")        # Limpar dataset_path_display
            )
        dataset_path, message = upload_dataset([], None, "start", dataset_name=dataset_name)
        if "already exists" in message:
            return (
                gr.update(value=message), 
                gr.update(value=None), 
                gr.update(visible=True),   # Manter botão visível
                gr.update(visible=False),
                gr.update(value="")        # Limpar dataset_path_display
            )
        return (
            gr.update(value=message), 
            dataset_path, 
            gr.update(visible=False), 
            gr.update(visible=True),
            gr.update(value=dataset_path)  # Atualizar dataset_path_display
        )
    
    

    with gr.Row(visible=True, elem_id="create_new_dataset_container") as create_new_container:
            with gr.Column():
                with gr.Row():
                    dataset_name_input = gr.Textbox(
                        label="Dataset Name",
                        placeholder="Enter your dataset name",
                        interactive=True
                    )
                create_dataset_button = gr.Button("Create Dataset", interactive=False)  # Inicialmente desativado
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                upload_files = gr.File(
                    label="Upload Images (.jpg, .png, .gif, .bmp, .webp), Videos (.mp4), Captions (.txt) or a ZIP archive",
                    file_types=[".jpg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".txt", ".zip"],
                    file_count="multiple",
                    type="filepath", 
                    interactive=True,
                    visible=False
                )
                
    # Função para habilitar/desabilitar o botão "Start New Dataset" com base na entrada
    def toggle_start_button(name):
        if name.strip():
            return gr.update(interactive=True)
        else:
            return gr.update(interactive=False)
    
    current_dataset_state = gr.State(None)
    
    dataset_name_input.change(
        fn=toggle_start_button, 
        inputs=dataset_name_input, 
        outputs=create_dataset_button
    )
    
    def handle_upload(files, current_dataset):
        updated_dataset, message = upload_dataset(files, current_dataset, "add")
        return updated_dataset, message, show_media(updated_dataset)
    
    
    # Container para selecionar dataset existente
    with gr.Row(visible=False, elem_id="select_existing_dataset_container") as select_existing_container:
        with gr.Column():
            existing_datasets = gr.Dropdown(
                choices=[],  # Inicialmente vazio; será atualizado dinamicamente
                label="Select Existing Dataset",
                interactive=True
            )
            # select_dataset_button = gr.Button("Select Dataset", interactive=False)  # Inicialmente desativado
    
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
    
    # Upload de arquivos and update gallery
    upload_files.upload(
        fn=lambda files, current_dataset: handle_upload(files, current_dataset),
        inputs=[upload_files, current_dataset_state],
        outputs=[current_dataset_state, upload_status, gallery],
        queue=True
    )
    
    # Função para lidar com a seleção de dataset existente e atualizar a galeria
    def handle_select_existing(selected_dataset):
        if selected_dataset:
            dataset_path = os.path.join(BASE_DATASET_DIR, selected_dataset)
            return dataset_path
        return ""
    
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
     
    # Evento para atualizar o caminho do dataset e a galeria ao selecionar um existente
    existing_datasets.change(
        fn=handle_select_existing,
        inputs=existing_datasets,
        outputs=[dataset_path]  # Atualizar dataset_path_display e gallery
    )
   
    dataset_option.change(
        fn=toggle_dataset_option,
        inputs=dataset_option,
        outputs=[create_new_container, select_existing_container, existing_datasets, dataset_name_input, upload_status, dataset_path, create_dataset_button, upload_files]
    )
    
    # Atualizar config path and output path
    def update_config_output_path(dataset_path):
        config_path = os.path.join(CONFIG_DIR, os.path.basename(dataset_path))
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(dataset_path))
        return config_path, output_path
    
     
    # Atualizar galeria quando o caminho do dataset muda
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
                value="adamw_optimizer",
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
                    train_button = gr.Button("Start Training")
                    with gr.Row():
                        with gr.Column(scale=1):
                            output = gr.Textbox(
                                label="Output Logs",
                                lines=20,
                                interactive=False,
                                elem_id="log_box"
                            )

    train_button.click(
        fn=train_model,
        inputs=[
            dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
            gradient_accumulation_steps, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar,
            num_ar_buckets, frame_buckets, gradient_clipping, warmup_steps, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes,
            activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
            video_clip_mode
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace", "."])
    