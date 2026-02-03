#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J cpp_full
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/cpp_full_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/cpp_full_%j.err
#SBATCH --gpus-per-node=4

set -e

module load cudatoolkit
module load cray-mpich
module load cmake

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp

# Clean and rebuild
rm -rf build_full
mkdir -p build_full
cd build_full

cmake .. -DCMAKE_BUILD_TYPE=Release -DPERLMUTTER=ON -DUSE_MPI=OFF
make -j16 full_bench

echo ""
echo "============================================"
echo "Running C++ Full System Benchmark"
echo "============================================"

./full_bench --block-size 512 --iters 5

echo ""
echo "============================================"
echo "Running with 1GB blocks"
echo "============================================"

./full_bench --block-size 1024 --iters 3

echo ""
echo "End: $(date)"
