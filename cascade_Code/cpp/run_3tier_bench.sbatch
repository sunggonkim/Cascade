#!/bin/bash
#SBATCH --job-name=cascade_3tier
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=logs/3tier_%j.out
#SBATCH --error=logs/3tier_%j.err

echo "=============================================="
echo "Cascade C++ 3-Tier Benchmark (GPU+SHM+Lustre)"
echo "=============================================="
echo "Date: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "Job ID: $SLURM_JOB_ID"

module load cudatoolkit/12.4

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp

LUSTRE_PATH="/pscratch/sd/s/sgkim/cascade_lustre"
BENCH="./cascade_bench"

echo ""
echo "=== GPU Info ==="
srun --nodes=1 --ntasks=1 nvidia-smi --query-gpu=name,memory.total --format=csv | head -5

echo ""
echo "=== Lustre Stripe Info ==="
lfs getstripe $LUSTRE_PATH 2>/dev/null || echo "Using default stripe"
lfs setstripe -c 4 -S 4M $LUSTRE_PATH 2>/dev/null || true

echo ""
echo "================================================================"
echo "  PART 1: GPU-Only Large Scale (up to 35GB per node)"
echo "================================================================"
echo "--- 20GB GPU test (fits in 40GB GPU memory) ---"
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 20000 --threads 32 --iterations 3 --gpu-only

echo ""
echo "--- 35GB GPU test (near GPU limit) ---"
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 35000 --threads 32 --iterations 2 --gpu-only

echo ""
echo "================================================================"
echo "  PART 2: SHM-Only Large Scale"
echo "================================================================"
echo "--- 20GB SHM test ---"
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 20000 --threads 32 --iterations 3

echo ""
echo "================================================================"
echo "  PART 3: Lustre Backend Test"
echo "================================================================"
echo "--- 10GB Lustre test ---"
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 10000 --threads 32 --iterations 3 \
    --lustre --lustre-path $LUSTRE_PATH

echo ""
echo "--- 50GB Lustre test (beyond GPU memory) ---"
rm -rf $LUSTRE_PATH/*
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 50000 --threads 32 --iterations 2 \
    --lustre --lustre-path $LUSTRE_PATH

echo ""
echo "================================================================"
echo "  PART 4: 4-Node Scaling with Large Data"
echo "================================================================"
echo "--- 4 Nodes × 20GB each = 80GB total ---"
srun --nodes=4 --ntasks=4 $BENCH --size 1024 --blocks 20000 --threads 32 --iterations 3 --gpu-only

echo ""
echo "================================================================"
echo "  PART 5: Full 3-Tier Cascade Test (GPU + SHM + Lustre)"
echo "================================================================"
echo "--- All 3 backends, 10GB each ---"
srun --nodes=1 --ntasks=1 $BENCH --size 1024 --blocks 10000 --threads 32 --iterations 3 \
    --lustre --lustre-path $LUSTRE_PATH

echo ""
echo "Complete at $(date)"
echo "=============================================="
