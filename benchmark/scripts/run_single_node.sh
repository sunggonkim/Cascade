#!/bin/bash
# benchmark/scripts/run_single_node.sh
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/benchmark_%j.out

module load python cudatoolkit

cd /pscratch/sd/s/sgkim/Skim-cascade

# Run all benchmarks
python -m benchmark.run_benchmark \
    --systems cascade,hdf5,lmcache \
    --workload all \
    --num_blocks 5000 \
    --data_path /pscratch/sd/s/sgkim/Skim-cascade/benchmark/data \
    --output /pscratch/sd/s/sgkim/Skim-cascade/benchmark/results

echo "Benchmark complete"
