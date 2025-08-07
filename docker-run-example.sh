#!/bin/bash

# Example Docker run commands for diffusion-pipe-ui with model persistence
# This script provides examples of how to run the container with proper volume mounts

echo "=== Diffusion Pipe UI Docker Run Examples ==="
echo ""

# Create directories if they don't exist
echo "Creating example directories..."
mkdir -p ./models
mkdir -p ./outputs
mkdir -p ./datasets
mkdir -p ./configs

echo "✓ Directories created: ./models, ./outputs, ./datasets, ./configs"
echo ""

echo "=== Example 1: Basic Run with Model Persistence ==="
echo "This will download models on first run and persist them between container restarts:"
echo ""
echo "docker run --gpus all -d \\"
echo "  -v \$(pwd)/models:/workspace/models \\"
echo "  -v \$(pwd)/outputs:/workspace/outputs \\"
echo "  -v \$(pwd)/datasets:/workspace/datasets \\"
echo "  -v \$(pwd)/configs:/workspace/configs \\"
echo "  -p 7860:7860 \\"
echo "  -p 8888:8888 \\"
echo "  -p 6006:6006 \\"
echo "  diffusion-pipe-ui"
echo ""

echo "=== Example 2: Skip Model Downloads (if you already have models) ==="
echo "Use this if you already have the models downloaded and want to skip re-downloading:"
echo ""
echo "docker run --gpus all -d \\"
echo "  -v \$(pwd)/models:/workspace/models \\"
echo "  -v \$(pwd)/outputs:/workspace/outputs \\"
echo "  -v \$(pwd)/datasets:/workspace/datasets \\"
echo "  -v \$(pwd)/configs:/workspace/configs \\"
echo "  -p 7860:7860 \\"
echo "  -p 8888:8888 \\"
echo "  -p 6006:6006 \\"
echo "  -e DOWNLOAD_MODELS=false \\"
echo "  diffusion-pipe-ui"
echo ""

echo "=== Example 3: Interactive Mode (for debugging) ==="
echo "Use this to see logs and interact with the container:"
echo ""
echo "docker run --gpus all -it \\"
echo "  -v \$(pwd)/models:/workspace/models \\"
echo "  -v \$(pwd)/outputs:/workspace/outputs \\"
echo "  -v \$(pwd)/datasets:/workspace/datasets \\"
echo "  -v \$(pwd)/configs:/workspace/configs \\"
echo "  -p 7860:7860 \\"
echo "  -p 8888:8888 \\"
echo "  -p 6006:6006 \\"
echo "  diffusion-pipe-ui"
echo ""

echo "=== Example 4: Windows Paths ==="
echo "For Windows users, replace paths with Windows-style paths:"
echo ""
echo "docker run --gpus all -d \\"
echo "  -v D:\\AI\\diffusion-pipe\\models:/workspace/models \\"
echo "  -v D:\\AI\\diffusion-pipe\\outputs:/workspace/outputs \\"
echo "  -v D:\\AI\\diffusion-pipe\\datasets:/workspace/datasets \\"
echo "  -v D:\\AI\\diffusion-pipe\\configs:/workspace/configs \\"
echo "  -p 7860:7860 \\"
echo "  -p 8888:8888 \\"
echo "  -p 6006:6006 \\"
echo "  diffusion-pipe-ui"
echo ""

echo "=== Port Information ==="
echo "• Port 7860: Gradio Web Interface (http://localhost:7860)"
echo "• Port 8888: Jupyter Lab (http://localhost:8888)"
echo "• Port 6006: TensorBoard (http://localhost:6006)"
echo ""

echo "=== Model Persistence ==="
echo "• Models are stored in ./models and persist between container restarts"
echo "• The container will check for existing models before downloading"
echo "• Set DOWNLOAD_MODELS=false to skip downloads if models already exist"
echo ""

echo "=== Troubleshooting ==="
echo "• If models aren't persisting, check that the volume mount paths are correct"
echo "• Use 'docker logs <container_id>' to see container logs"
echo "• Use 'docker exec -it <container_id> /bin/bash' to access the container"
echo ""

echo "=== Quick Start ==="
echo "To start with model persistence, run:"
echo "docker run --gpus all -d -v \$(pwd)/models:/workspace/models -v \$(pwd)/outputs:/workspace/outputs -v \$(pwd)/datasets:/workspace/datasets -v \$(pwd)/configs:/workspace/configs -p 7860:7860 -p 8888:8888 -p 6006:6006 diffusion-pipe-ui" 