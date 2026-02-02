#!/bin/bash
# benchmark/scripts/generate_shared_data.sh
# Generate shared KV cache data for all benchmark systems
#
# Usage:
#   sbatch benchmark/scripts/generate_shared_data.sh
#
#SBATCH -A m4431
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -o benchmark/logs/generate_data_%j.out
#SBATCH -e benchmark/logs/generate_data_%j.err
#SBATCH -J cascade_data_gen

set -e

echo "=========================================="
echo "Cascade Benchmark Data Generator"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Environment
module load python
cd /pscratch/sd/s/sgkim/Skim-cascade

# Create log directory
mkdir -p benchmark/logs

# Configuration
SIZE_GB=${SIZE_GB:-500}          # Total data size in GB
NUM_PREFIXES=${NUM_PREFIXES:-100}  # Number of unique system prompts
SESSIONS_PER_PREFIX=${SESSIONS_PER_PREFIX:-50}  # Sessions sharing each prefix
OUTPUT_DIR="/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data"

echo ""
echo "Configuration:"
echo "  Size: ${SIZE_GB} GB"
echo "  Prefixes: ${NUM_PREFIXES}"
echo "  Sessions per prefix: ${SESSIONS_PER_PREFIX}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Run generator
python -m benchmark.data_generator_optimized \
    --size_gb $SIZE_GB \
    --output $OUTPUT_DIR \
    --num_prefixes $NUM_PREFIXES \
    --sessions_per_prefix $SESSIONS_PER_PREFIX

echo ""
echo "=========================================="
echo "Data generation complete!"
echo "=========================================="

# Verify data
echo ""
echo "Verifying data integrity..."
python -c "
from benchmark.shared_data import verify_data_integrity
verify_data_integrity('$OUTPUT_DIR', sample_size=100)
"

# Show disk usage
echo ""
echo "Disk usage:"
du -sh $OUTPUT_DIR
du -sh $OUTPUT_DIR/*

echo ""
echo "Done at $(date)"
