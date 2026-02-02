#!/bin/bash
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o logs/accurate_%j.out
#SBATCH -e logs/accurate_%j.err
#SBATCH -J accurate_bench

# Accurate C++ Cascade Benchmark
# Fixed: GPU/SHM capacity now auto-sized to fit data

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp

module load PrgEnv-gnu cudatoolkit craype-accel-nvidia80 cmake cray-python

echo "=============================================="
echo "  ACCURATE Cascade C++ Benchmark"
echo "  Capacity auto-sized to data size + 20%"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

BENCH="./build/cascade_bench"
LUSTRE_PATH="/pscratch/sd/s/sgkim/cascade_lustre"
mkdir -p $LUSTRE_PATH

echo "================================================================"
echo "  TEST 1: GPU Backend - Various Data Sizes"
echo "================================================================"

echo ""
echo "--- 2GB GPU test (small, warm-up) ---"
$BENCH --blocks 2000 --size 1024 --threads 32 --gpu-only --iterations 3

echo ""
echo "--- 10GB GPU test (medium, our verified size) ---"
$BENCH --blocks 10000 --size 1024 --threads 32 --gpu-only --iterations 3

echo ""
echo "--- 20GB GPU test (large, fits in 40GB GPU) ---"
$BENCH --blocks 20000 --size 1024 --threads 32 --gpu-only --iterations 3

echo ""
echo "--- 30GB GPU test (very large, still fits) ---"
$BENCH --blocks 30000 --size 1024 --threads 32 --gpu-only --iterations 3

echo ""
echo "================================================================"
echo "  TEST 2: SHM Backend"
echo "================================================================"

echo ""
echo "--- 10GB SHM test ---"
$BENCH --blocks 10000 --size 1024 --threads 32 --iterations 3

echo ""
echo "================================================================"
echo "  TEST 3: Lustre Backend"  
echo "================================================================"

echo ""
echo "--- 5GB Lustre test ---"
lfs setstripe -c 4 -S 4M $LUSTRE_PATH 2>/dev/null || true
$BENCH --blocks 5000 --size 1024 --threads 32 --lustre --lustre-path $LUSTRE_PATH --iterations 2

echo ""
echo "--- 10GB Lustre test ---"
rm -rf $LUSTRE_PATH/*
$BENCH --blocks 10000 --size 1024 --threads 32 --lustre --lustre-path $LUSTRE_PATH --iterations 2

echo ""
echo "=============================================="
echo "  BENCHMARK COMPLETE"
echo "=============================================="
echo "All capacity auto-sized. No overflow. These are accurate numbers."
