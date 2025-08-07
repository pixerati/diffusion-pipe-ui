#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"true"}  # Default if not set
DOWNLOAD_BF16=${DOWNLOAD_BF16:-"false"}  # Default if not set
REPO_DIR=${REPO_DIR:-"/workspace/diffusion-pipe"}

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS and DOWNLOAD_BF16 is: $DOWNLOAD_BF16"

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv

if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    echo "Installing CUDA nvcc..."
    conda install -y -c nvidia cuda-nvcc --override-channels

    echo "Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r $REPO_DIR/requirements.txt

    # Create marker file
    touch "$INIT_MARKER"
    echo "Initialization complete."
else
    echo "Container already initialized. Skipping first-time setup."
fi

# Model download logic - runs on every container start if DOWNLOAD_MODELS=true
if [ "$DOWNLOAD_MODELS" = "true" ]; then
    echo "DOWNLOAD_MODELS is true, checking for existing models..."
    MODEL_DIR="/workspace/models"
    mkdir -p "$MODEL_DIR"

    # Function to check if a model file/directory exists and is not empty
    check_model_exists() {
        local model_path="$1"
        if [ -d "$model_path" ]; then
            # For directories, check if it contains files
            if [ "$(ls -A "$model_path" 2>/dev/null)" ]; then
                return 0  # Directory exists and is not empty
            else
                return 1  # Directory exists but is empty
            fi
        elif [ -f "$model_path" ]; then
            # For files, check if file exists and has size > 0
            if [ -s "$model_path" ]; then
                return 0  # File exists and has content
            else
                return 1  # File exists but is empty
            fi
        else
            return 1  # Path doesn't exist
        fi
    }

    # Function to download model with better error handling
    download_model() {
        local model_name="$1"
        local model_path="$2"
        local download_args="$3"
        
        echo "Checking for model: $model_name at $model_path"
        if check_model_exists "$model_path"; then
            echo "✓ Model $model_name already exists at $model_path, skipping download."
        else
            echo "✗ Model $model_name not found or empty at $model_path, downloading..."
            if huggingface-cli download $download_args; then
                echo "✓ Successfully downloaded $model_name"
            else
                echo "✗ Failed to download $model_name"
                return 1
            fi
        fi
    }

    # Download llava-llama-3-8b-text-encoder-tokenizer
    download_model "llava-llama-3-8b-text-encoder-tokenizer" \
        "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer" \
        "Kijai/llava-llama-3-8b-text-encoder-tokenizer --local-dir ${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer"

    # Download hunyuan_video_720_cfgdistill_fp8_e4m3fn
    download_model "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" \
        "${MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" \
        "Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors --local-dir ${MODEL_DIR}"

    # Download hunyuan_video_720_cfgdistill_bf16 (conditional)
    if [ "${DOWNLOAD_BF16}" = "true" ]; then
        download_model "hunyuan_video_720_cfgdistill_bf16.safetensors" \
            "${MODEL_DIR}/hunyuan_video_720_cfgdistill_bf16.safetensors" \
            "Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_bf16.safetensors --local-dir ${MODEL_DIR}"
    else
        echo "Skipping hunyuan_video_720_cfgdistill_bf16.safetensors download (DOWNLOAD_BF16=false)"
    fi

    # Download hunyuan_video_vae_fp32
    download_model "hunyuan_video_vae_fp32.safetensors" \
        "${MODEL_DIR}/hunyuan_video_vae_fp32.safetensors" \
        "Kijai/HunyuanVideo_comfy hunyuan_video_vae_fp32.safetensors --local-dir ${MODEL_DIR}"

    # Download hunyuan_video_vae_bf16
    download_model "hunyuan_video_vae_bf16.safetensors" \
        "${MODEL_DIR}/hunyuan_video_vae_bf16.safetensors" \
        "Kijai/HunyuanVideo_comfy hunyuan_video_vae_bf16.safetensors --local-dir ${MODEL_DIR}"

    # Download clip-vit-large-patch14
    download_model "clip-vit-large-patch14" \
        "${MODEL_DIR}/clip-vit-large-patch14" \
        "openai/clip-vit-large-patch14 --local-dir ${MODEL_DIR}/clip-vit-large-patch14"

    # Download WAN 2.1 T2V 14B model
    echo "Downloading WAN 2.1 models..."
    download_model "Wan2.1-T2V-14B" \
        "${MODEL_DIR}/Wan2.1-T2V-14B" \
        "Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR}/Wan2.1-T2V-14B"

    # Download WAN 2.2 models
    echo "Downloading WAN 2.2 models..."
    
    # Download WAN 2.2 TI2V 5B model (only supported for training)
    download_model "Wan2.2-TI2V-5B" \
        "${MODEL_DIR}/Wan2.2-TI2V-5B" \
        "Wan-AI/Wan2.2-TI2V-5B --local-dir ${MODEL_DIR}/Wan2.2-TI2V-5B"

    echo "Model download check complete."
else
    echo "DOWNLOAD_MODELS is false, skipping model downloads."
fi

echo "Adding environment variables"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/submodules/HunyuanVideo:$PYTHONPATH"
export PATH="$REPO_DIR/configs:$PATH"
export PATH="$REPO_DIR:$PATH"

echo $PATH
echo $PYTHONPATH

cd /workspace/diffusion-pipe

# Use conda python instead of system python
echo "Starting Gradio interface..."
python gradio_interface.py &

echo "Starting Tensorboard interface..."
$CONDA_DIR/bin/conda run -n pyenv tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &

wait