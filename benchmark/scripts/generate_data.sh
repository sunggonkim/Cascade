#!/bin/bash
# benchmark/scripts/generate_data.sh
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/generate_data_%j.out

module load python cudatoolkit

cd /pscratch/sd/s/sgkim/Skim-cascade

# Generate 500GB of benchmark data
python -m benchmark.data_generator \
    --size_gb 500 \
    --output /pscratch/sd/s/sgkim/Skim-cascade/benchmark/data \
    --block_tokens 256

echo "Data generation complete"
