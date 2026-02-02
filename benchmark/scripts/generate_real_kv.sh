#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o logs/generate_real_%j.out
#SBATCH -e logs/generate_real_%j.err
#SBATCH -J cascade_kv_gen

###############################################################################
# REAL KV Cache Generation for Cascade Benchmarks
# 
# Strategy:
#   1. Download MLPerf OpenOrca dataset (if not exists)
#   2. Run C++ MPI aggregator for fast block generation
#   3. (Optional) Run Python vLLM extractor for real model inference
#
# Output:
#   $SCRATCH/cascade_kv_cache/  (~500GB aggregated KV cache)
###############################################################################

set -e

# Configuration
export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export OUTPUT_DIR=$SCRATCH/cascade_kv_cache
export DATASET_DIR=$SCRATCH/mlperf_data

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich
module load cudatoolkit

echo "============================================"
echo "Cascade KV Cache Generation"
echo "============================================"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Output: $OUTPUT_DIR"
echo "============================================"

cd $PROJECT_DIR/benchmark/kv_extractor

###############################################################################
# Step 1: Download OpenOrca dataset (if not exists)
###############################################################################
if [ ! -f "$DATASET_DIR/open_orca_gpt4_tokenized_llama.pkl" ]; then
    echo "[Step 1] Downloading OpenOrca dataset..."
    mkdir -p $DATASET_DIR
    
    # Try MLCommons R2 downloader first
    if command -v curl &> /dev/null; then
        echo "Downloading from MLCommons R2..."
        cd $DATASET_DIR
        bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
            https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri || true
        cd $PROJECT_DIR/benchmark/kv_extractor
    fi
    
    # Fallback: process from HuggingFace
    if [ ! -f "$DATASET_DIR/open_orca_gpt4_tokenized_llama.pkl" ]; then
        echo "Downloading from HuggingFace..."
        python3 -c "
from datasets import load_dataset
import pandas as pd
import pickle

print('Loading Open-Orca/OpenOrca...')
ds = load_dataset('Open-Orca/OpenOrca', split='train[:24576]')

data = {
    'prompt': [ex['question'] for ex in ds],
    'system_prompt': [ex.get('system_prompt', '') for ex in ds],
}

df = pd.DataFrame(data)
df.to_pickle('$DATASET_DIR/open_orca_gpt4_tokenized_llama.pkl')
print(f'Saved {len(df)} samples')
"
    fi
else
    echo "[Step 1] Dataset already exists, skipping download"
fi

###############################################################################
# Step 2: Build C++ aggregator
###############################################################################
echo "[Step 2] Building C++ KV aggregator..."
CC -O3 -fopenmp -std=c++17 -o kv_aggregator kv_aggregator.cpp -lcrypto 2>/dev/null || \
    g++ -O3 -fopenmp -std=c++17 -o kv_aggregator kv_aggregator.cpp -lcrypto -lmpi

###############################################################################
# Step 3: Generate aggregated KV cache (C++ MPI - FAST)
###############################################################################
echo "[Step 3] Generating aggregated KV cache with MPI..."
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Build MPI version (use mpicxx to avoid CUDA dependency)
module load cray-mpich
cd $PROJECT_DIR/benchmark/kv_extractor
mpicxx -O3 -fopenmp -std=c++17 -o kv_aggregator kv_aggregator.cpp -lcrypto

# Calculate sessions to generate ~500GB of data
# Block size: ~164MB (key + value)
# 500GB / 164MB = ~3000 blocks
# With 32 tasks, ~100 sessions per rank, 4 blocks per session
SESSIONS_PER_RANK=$((3000 / $SLURM_NTASKS / 4))
SESSIONS_PER_RANK=$((SESSIONS_PER_RANK > 100 ? SESSIONS_PER_RANK : 100))

echo "Sessions per rank: $SESSIONS_PER_RANK"
echo "Expected output: ~$((SESSIONS_PER_RANK * SLURM_NTASKS * 4 * 164 / 1024)) GB"

time srun --ntasks=$SLURM_NTASKS ./kv_aggregator \
    --output $OUTPUT_DIR \
    --sessions $SESSIONS_PER_RANK \
    --blocks 4

echo "============================================"
echo "C++ generation complete!"
echo "============================================"

# Show output stats
echo "Output directory contents:"
du -sh $OUTPUT_DIR
ls -la $OUTPUT_DIR/

###############################################################################
# Step 4: (Optional) Create global index for all ranks
###############################################################################
echo "[Step 4] Creating global index..."
python3 << 'EOF'
import os
import json
import struct
from pathlib import Path
from collections import defaultdict

output_dir = Path(os.environ['OUTPUT_DIR'])
global_index = defaultdict(list)  # block_id -> [(rank, file, offset)]

for rank_dir in sorted(output_dir.glob('rank_*')):
    rank = int(rank_dir.name.split('_')[1])
    
    for agg_file in sorted(rank_dir.glob('agg_*.bin')):
        file_id = agg_file.stem
        
        with open(agg_file, 'rb') as f:
            # Read header
            magic = f.read(8)
            if magic != b'CASKV001':
                continue
            
            block_count = struct.unpack('<I', f.read(4))[0]
            
            # Seek to index
            f.seek(-8, 2)
            index_offset = struct.unpack('<Q', f.read(8))[0]
            
            # Read index
            f.seek(index_offset)
            for _ in range(block_count):
                block_id = f.read(32).decode('utf-8')
                offset, size = struct.unpack('<QQ', f.read(16))
                
                global_index[block_id].append({
                    'rank': rank,
                    'file': str(agg_file.relative_to(output_dir)),
                    'offset': offset,
                    'size': size
                })

# Write global index
index_path = output_dir / 'global_index.json'
with open(index_path, 'w') as f:
    json.dump({
        'total_unique_blocks': len(global_index),
        'blocks': dict(global_index)
    }, f, indent=2)

print(f"Global index created: {len(global_index)} unique blocks")
EOF

echo "============================================"
echo "KV Cache Generation Complete!"
echo "============================================"
echo "Output: $OUTPUT_DIR"
du -sh $OUTPUT_DIR
cat $OUTPUT_DIR/summary.json
