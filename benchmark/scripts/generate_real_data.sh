#!/bin/bash
# benchmark/scripts/generate_real_data.sh
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/generate_real_%j.out

module load python cudatoolkit
module load pytorch/2.1.0

cd /pscratch/sd/s/sgkim/Skim-cascade

# Hugging Face token (if needed)
# export HF_TOKEN="your_token_here"
export HF_HOME=/pscratch/sd/s/sgkim/Skim-cascade/benchmark/hf_cache

# Generate real KV cache data from LLaMA-2-70B
python -m benchmark.data_generator_real \
    --model meta-llama/Llama-2-70b-hf \
    --datasets openorca,cnn_dailymail,sharegpt \
    --num_samples 10000 \
    --output /pscratch/sd/s/sgkim/Skim-cascade/benchmark/data_real \
    --tp 4 \
    --block_size 256 \
    --shared_prefix

echo "Real data generation complete"
