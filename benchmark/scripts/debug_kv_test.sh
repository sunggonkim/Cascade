#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/debug_kv_%j.out
#SBATCH -e logs/debug_kv_%j.err
#SBATCH -J debug_kv

###############################################################################
# Debug test for KV Cache Aggregator
# Quick test with 2 nodes, 8 MPI ranks
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export OUTPUT_DIR=$SCRATCH/cascade_kv_cache_debug

cd $PROJECT_DIR/benchmark/kv_extractor
mkdir -p logs

echo "============================================"
echo "Debug KV Cache Generation Test"
echo "============================================"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Build MPI version
echo "Building MPI version..."
# Use mpicxx instead of CC to avoid CUDA dependency
module load cray-mpich
mpicxx -O3 -fopenmp -std=c++17 -o kv_aggregator kv_aggregator.cpp -lcrypto

# Clean output
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Run with MPI - small test (10 sessions x 4 blocks = 40 blocks per rank)
echo "Running MPI test..."
time srun --ntasks=$SLURM_NTASKS ./kv_aggregator \
    --output $OUTPUT_DIR \
    --sessions 10 \
    --blocks 4

echo ""
echo "============================================"
echo "Test complete!"
echo "============================================"

# Show results
echo "Output:"
du -sh $OUTPUT_DIR
ls -la $OUTPUT_DIR/
cat $OUTPUT_DIR/summary.json
