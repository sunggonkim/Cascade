#!/bin/bash
#SBATCH --job-name=cascade_long
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=logs/cascade_long_%j.out
#SBATCH --error=logs/cascade_long_%j.err

echo "=============================================="
echo "Cascade C++ Long Benchmark - SC26 Evaluation"
echo "=============================================="
echo "Date: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "Job ID: $SLURM_JOB_ID"

module load cudatoolkit/12.4

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp

echo ""
echo "=== GPU Info ==="
srun --nodes=1 --ntasks=1 nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv

BENCH="./cascade_bench"

echo ""
echo "================================================================"
echo "  PART 1: Single Node Block Size Sweep (2GB total data each)"
echo "================================================================"
for size in 128 256 512 1024 2048 4096; do
    blocks=$((2048 * 1024 / size))
    echo ""
    echo "--- Block size: ${size}KB, Blocks: $blocks ---"
    srun --nodes=1 --ntasks=1 $BENCH --size $size --blocks $blocks --threads 32 --iterations 5 --gpu-only
done

echo ""
echo "================================================================"
echo "  PART 2: Thread Scaling (1MB blocks, 2GB data)"
echo "================================================================"
for threads in 1 2 4 8 16 32 64; do
    echo ""
    echo "--- Threads: $threads ---"
    srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 2000 --threads $threads --iterations 5 --gpu-only
done

echo ""
echo "================================================================"
echo "  PART 3: Multi-Node Scaling (1MB blocks, 2GB per node)"
echo "================================================================"
for nodes in 1 2 4; do
    echo ""
    echo "--- Nodes: $nodes ---"
    srun --nodes=$nodes --ntasks=$nodes $BENCH --size 1024 --blocks 2000 --threads 32 --iterations 5 --gpu-only
done

echo ""
echo "================================================================"
echo "  PART 4: Large Data Test (10GB per node)"
echo "================================================================"
echo "--- Single Node, 10GB ---"
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 10000 --threads 32 --iterations 3 --gpu-only

echo ""
echo "--- 4 Nodes, 10GB each (40GB total) ---"
srun --nodes=4 --ntasks=4 $BENCH --size 1024 --blocks 10000 --threads 32 --iterations 3 --gpu-only

echo ""
echo "================================================================"
echo "  PART 5: SHM Backend Test"
echo "================================================================"
echo "--- Single Node, 2GB SHM ---"
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 2000 --threads 32 --iterations 5

echo ""
echo "Complete at $(date)"
echo "=============================================="
