#!/bin/bash
# benchmark/scripts/run_debug.sh
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/benchmark_debug_%j.out

module load python cudatoolkit

cd /pscratch/sd/s/sgkim/Skim-cascade

# Ensure log directory exists
mkdir -p benchmark/logs

# Run all benchmarks (including stubs) for 5 schemes
# Using systems="all" to cover cascade, hdf5, lmcache, redis, pdc
python -m benchmark.run_benchmark \
    --systems all \
    --workload all \
    --num_blocks 1000 \
    --data_path /pscratch/sd/s/sgkim/Skim-cascade/benchmark/data \
    --output /pscratch/sd/s/sgkim/Skim-cascade/benchmark/results

echo "Debug benchmark complete"
