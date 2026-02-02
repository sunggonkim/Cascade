#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/generate_kv_%j.out
#SBATCH -e logs/generate_kv_%j.err
#SBATCH -J kv_gen

###############################################################################
# KV Cache Generation for Cascade Benchmarks
# 
# Uses C++ MPI aggregator for fast parallel generation
# Output: $SCRATCH/cascade_kv_cache/
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export OUTPUT_DIR=$SCRATCH/cascade_kv_cache

cd $PROJECT_DIR/benchmark/kv_extractor
mkdir -p logs

echo "============================================"
echo "Cascade KV Cache Generation"
echo "============================================"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Build MPI version
echo "[Step 1] Building C++ MPI aggregator..."
module load cray-mpich
mpicxx -O3 -fopenmp -std=c++17 -o kv_aggregator kv_aggregator.cpp -lcrypto

# Clean and create output
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Calculate sessions for ~500GB target
# Block size: ~164MB (82MB key + 82MB value)
# 500GB = 500 * 1024 MB / 164 MB = ~3100 blocks
# With 16 tasks and 4 blocks per session: 3100 / 16 / 4 = ~49 sessions
SESSIONS_PER_RANK=50

echo "[Step 2] Generating KV cache..."
echo "Sessions per rank: $SESSIONS_PER_RANK"
echo "Blocks per session: 4"
echo "Expected blocks: $((SLURM_NTASKS * SESSIONS_PER_RANK * 4))"
echo "Expected size: $((SLURM_NTASKS * SESSIONS_PER_RANK * 4 * 164 / 1024)) GB"

time srun --ntasks=$SLURM_NTASKS ./kv_aggregator \
    --output $OUTPUT_DIR \
    --sessions $SESSIONS_PER_RANK \
    --blocks 4

echo ""
echo "============================================"
echo "Generation complete!"
echo "============================================"

# Show results
du -sh $OUTPUT_DIR
ls -la $OUTPUT_DIR/
cat $OUTPUT_DIR/summary.json

echo ""
echo "[Step 3] Creating global index..."
python3 << 'PYEOF'
import os
import json
import struct
from pathlib import Path
from collections import defaultdict

output_dir = Path(os.environ['OUTPUT_DIR'])
global_index = {}
total_blocks = 0

for rank_dir in sorted(output_dir.glob('rank_*')):
    rank = int(rank_dir.name.split('_')[1])
    
    for agg_file in sorted(rank_dir.glob('agg_*.bin')):
        try:
            with open(agg_file, 'rb') as f:
                magic = f.read(8)
                if magic != b'CASKV001':
                    continue
                
                block_count = struct.unpack('<I', f.read(4))[0]
                
                f.seek(-8, 2)
                index_offset = struct.unpack('<Q', f.read(8))[0]
                
                f.seek(index_offset)
                for _ in range(block_count):
                    block_id = f.read(32).decode('utf-8')
                    offset, size = struct.unpack('<QQ', f.read(16))
                    
                    if block_id not in global_index:
                        global_index[block_id] = {
                            'file': str(agg_file.relative_to(output_dir)),
                            'offset': offset,
                            'size': size
                        }
                        total_blocks += 1
        except Exception as e:
            print(f"Warning: {agg_file}: {e}")

index_path = output_dir / 'global_index.json'
with open(index_path, 'w') as f:
    json.dump({
        'total_unique_blocks': total_blocks,
        'blocks': global_index
    }, f)

print(f"Global index: {total_blocks} unique blocks")
PYEOF

echo "============================================"
echo "All done!"
echo "============================================"
