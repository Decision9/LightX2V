git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
conda create -n lightx2v python=3.11 -y
conda activate lightx2v
pip install -v -e .

git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="80" MAX_JOBS=8 pip install . --no-build-isolation

git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
conda install -c nvidia cuda-nvcc=12.8 cuda-cudart=12.8 cuda-toolkit=12.8
CUDA_ARCHITECTURES="8.6" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 pip install -v . --no-build-isolation
pip install sgl-kernel==0.3.8

For a list of CUDA-capable GPUs, see the [NVIDIA CUDA GPU documentation](https://developer.nvidia.com/cuda/gpus).
