# benchmark/data_generator_optimized.py
"""
Optimized KV Cache Data Generator for HPC Benchmarks.

Key optimizations:
1. Aggregated binary files (not per-file) - avoids Lustre metadata overhead
2. Lustre striping for large files
3. Memory-mapped writing for efficiency
4. Content-addressed block IDs (SHA-256)
5. Single index file for fast lookup

Usage:
    # On login node (small test):
    python -m benchmark.data_generator_optimized --size_gb 10 --output benchmark/data

    # On compute node (full generation):
    srun -N1 -n32 python -m benchmark.data_generator_optimized --size_gb 500 --output benchmark/data
"""
import os
import sys
import json
import hashlib
import argparse
import struct
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import subprocess

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from benchmark.config import LLAMA_CONFIG, BENCHMARK_CONFIG
except ImportError:
    # Fallback for direct execution
    @dataclass
    class LLaMAConfig:
        num_layers: int = 80
        num_kv_heads: int = 8
        head_dim: int = 128
        dtype: str = "float16"
        
        @property
        def kv_size_per_token(self) -> int:
            return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2
    
    LLAMA_CONFIG = LLaMAConfig()
    BENCHMARK_CONFIG = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataGenConfig:
    """Data generation configuration"""
    # Output
    output_dir: Path = Path("/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data")
    
    # KV Block dimensions (LLaMA-70B GQA)
    num_layers: int = 80
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"
    
    # Block configuration
    block_size_tokens: int = 256  # Tokens per block
    
    # Aggregation (HPC optimization)
    blocks_per_aggregate: int = 128  # ~10GB per aggregate file
    lustre_stripe_count: int = 16
    lustre_stripe_size: str = "4m"
    
    # Prefix sharing scenario
    num_prefixes: int = 100         # Number of unique system prompts
    prefix_length_tokens: int = 2048  # Tokens per prefix
    sessions_per_prefix: int = 50   # Sessions sharing each prefix
    
    # Unique blocks
    num_unique_sessions: int = 1000
    unique_tokens_per_session: int = 2048
    
    @property
    def block_shape(self) -> Tuple[int, ...]:
        """Shape of single KV block (key or value)"""
        return (self.block_size_tokens, self.num_layers, self.num_kv_heads, self.head_dim)
    
    @property
    def block_size_bytes(self) -> int:
        """Size of one complete KV block (key + value)"""
        single_tensor_bytes = np.prod(self.block_shape) * 2  # float16
        return single_tensor_bytes * 2  # key + value
    
    @property
    def blocks_per_prefix(self) -> int:
        return self.prefix_length_tokens // self.block_size_tokens


# =============================================================================
# Block Metadata
# =============================================================================

@dataclass
class BlockInfo:
    """Metadata for a single KV block"""
    block_id: str           # Content hash (SHA-256[:32])
    block_idx: int          # Global block index
    is_prefix: bool         # Is shared prefix block
    prefix_id: int          # Prefix group ID (-1 if not prefix)
    token_offset: int       # Starting token position
    # Location in aggregated storage
    agg_file_id: int        # Which aggregate file
    agg_offset: int         # Byte offset within file
    size_bytes: int         # Block size


@dataclass
class DatasetIndex:
    """Complete index for the dataset"""
    # Config
    config: Dict
    
    # Block index: block_id -> BlockInfo
    blocks: Dict[str, Dict]  # Serializable version
    
    # Prefix groups: prefix_id -> list of block_ids
    prefix_groups: Dict[int, List[str]]
    
    # Session mappings: session_id -> {prefix_blocks, unique_blocks}
    sessions: Dict[str, Dict]
    
    # Statistics
    stats: Dict
    
    def save(self, path: Path):
        """Save index to file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'DatasetIndex':
        """Load index from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# Aggregated File Writer
# =============================================================================

class AggregatedWriter:
    """
    Write KV blocks to aggregated binary files.
    
    File format:
        [Block 0][Block 1]...[Block N]
    
    Each block:
        [key_data: float16 array][value_data: float16 array]
    
    All blocks have fixed size, so offset = block_idx * block_size
    """
    
    def __init__(self, output_dir: Path, config: DataGenConfig):
        self.output_dir = Path(output_dir)
        self.agg_dir = self.output_dir / "aggregated"
        self.agg_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.blocks_per_file = config.blocks_per_aggregate
        self.block_size = config.block_size_bytes
        
        self.current_file_id = 0
        self.current_file = None
        self.current_block_count = 0
        
        self._setup_lustre_striping()
    
    def _setup_lustre_striping(self):
        """Configure Lustre striping for aggregate directory"""
        try:
            subprocess.run([
                "lfs", "setstripe",
                "-c", str(self.config.lustre_stripe_count),
                "-S", self.config.lustre_stripe_size,
                str(self.agg_dir)
            ], capture_output=True, check=False)
            print(f"[Writer] Lustre striping configured: -c {self.config.lustre_stripe_count} -S {self.config.lustre_stripe_size}")
        except FileNotFoundError:
            print("[Writer] lfs not available (not on Lustre)")
    
    def _open_new_file(self):
        """Open a new aggregate file"""
        if self.current_file:
            self.current_file.close()
        
        file_path = self.agg_dir / f"agg_{self.current_file_id:06d}.bin"
        self.current_file = open(file_path, 'wb')
        self.current_block_count = 0
    
    def write_block(self, key_data: np.ndarray, value_data: np.ndarray) -> Tuple[int, int]:
        """
        Write a block and return (file_id, offset).
        """
        # Open new file if needed
        if self.current_file is None or self.current_block_count >= self.blocks_per_file:
            if self.current_file:
                self.current_file_id += 1
            self._open_new_file()
        
        # Record position
        file_id = self.current_file_id
        offset = self.current_block_count * self.block_size
        
        # Write data (contiguous binary)
        self.current_file.write(key_data.tobytes())
        self.current_file.write(value_data.tobytes())
        
        self.current_block_count += 1
        
        return file_id, offset
    
    def close(self):
        """Close current file"""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
    
    @property
    def num_files(self) -> int:
        return self.current_file_id + 1


# =============================================================================
# Data Generator
# =============================================================================

class OptimizedDataGenerator:
    """
    Generate KV cache benchmark data in HPC-optimized format.
    
    Data structure:
    output_dir/
    ├── aggregated/
    │   ├── agg_000000.bin    # Multiple blocks per file
    │   ├── agg_000001.bin
    │   └── ...
    ├── index.json            # Block ID -> location mapping
    ├── index.pkl             # Pickle for faster loading
    └── metadata.json         # Dataset statistics
    """
    
    def __init__(self, config: DataGenConfig = None):
        self.config = config or DataGenConfig()
        self.blocks: List[BlockInfo] = []
        self.prefix_groups: Dict[int, List[str]] = {}
        self.sessions: Dict[str, Dict] = {}
    
    def _compute_block_id(self, key_data: np.ndarray, value_data: np.ndarray) -> str:
        """Compute content-addressed block ID"""
        hasher = hashlib.sha256()
        hasher.update(key_data.tobytes())
        hasher.update(value_data.tobytes())
        return hasher.hexdigest()[:32]
    
    def _generate_block_data(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate deterministic KV block data"""
        rng = np.random.RandomState(seed)
        shape = self.config.block_shape
        
        # Generate realistic KV cache values (small magnitude, normally distributed)
        key_data = (rng.randn(*shape) * 0.1).astype(np.float16)
        value_data = (rng.randn(*shape) * 0.1).astype(np.float16)
        
        return key_data, value_data
    
    def generate(self, size_gb: float = 100.0):
        """Generate complete dataset"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("Optimized KV Cache Data Generator")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print(f"Target size: {size_gb:.1f} GB")
        print(f"Block size: {self.config.block_size_tokens} tokens = {self.config.block_size_bytes / 1024 / 1024:.2f} MB")
        print(f"Blocks per aggregate file: {self.config.blocks_per_aggregate}")
        print("=" * 70)
        
        writer = AggregatedWriter(output_dir, self.config)
        start_time = time.time()
        
        # 1. Generate prefix blocks
        print("\n[1/3] Generating prefix blocks...")
        prefix_blocks = self._generate_prefixes(writer)
        
        # 2. Generate unique blocks to fill remaining space
        print("\n[2/3] Generating unique blocks...")
        prefix_size = len(prefix_blocks) * self.config.block_size_bytes / 1024**3
        remaining_gb = size_gb - prefix_size
        unique_blocks = self._generate_unique_blocks(writer, remaining_gb, len(prefix_blocks))
        
        # 3. Generate session mappings
        print("\n[3/3] Generating session mappings...")
        self._generate_sessions(prefix_blocks, unique_blocks)
        
        writer.close()
        
        # Combine all blocks
        all_blocks = prefix_blocks + unique_blocks
        self.blocks = all_blocks
        
        # Save index
        self._save_index(output_dir, all_blocks, writer.num_files)
        
        elapsed = time.time() - start_time
        total_size = len(all_blocks) * self.config.block_size_bytes / 1024**3
        
        print("\n" + "=" * 70)
        print("Generation Complete!")
        print("=" * 70)
        print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Total blocks: {len(all_blocks):,}")
        print(f"  Prefix blocks: {len(prefix_blocks):,}")
        print(f"  Unique blocks: {len(unique_blocks):,}")
        print(f"Total size: {total_size:.2f} GB")
        print(f"Aggregate files: {writer.num_files}")
        print(f"Sessions: {len(self.sessions):,}")
        print(f"Output: {output_dir}")
        print("=" * 70)
    
    def _generate_prefixes(self, writer: AggregatedWriter) -> List[BlockInfo]:
        """Generate shared prefix blocks"""
        blocks = []
        blocks_per_prefix = self.config.blocks_per_prefix
        
        total_prefix_blocks = self.config.num_prefixes * blocks_per_prefix
        print(f"  Prefixes: {self.config.num_prefixes}")
        print(f"  Blocks per prefix: {blocks_per_prefix}")
        print(f"  Total prefix blocks: {total_prefix_blocks}")
        
        for prefix_id in range(self.config.num_prefixes):
            self.prefix_groups[prefix_id] = []
            
            for block_idx in range(blocks_per_prefix):
                global_idx = prefix_id * blocks_per_prefix + block_idx
                seed = prefix_id * 10000 + block_idx  # Deterministic seed
                
                # Generate data
                key_data, value_data = self._generate_block_data(seed)
                block_id = self._compute_block_id(key_data, value_data)
                
                # Write to aggregate file
                file_id, offset = writer.write_block(key_data, value_data)
                
                # Create metadata
                info = BlockInfo(
                    block_id=block_id,
                    block_idx=global_idx,
                    is_prefix=True,
                    prefix_id=prefix_id,
                    token_offset=global_idx * self.config.block_size_tokens,
                    agg_file_id=file_id,
                    agg_offset=offset,
                    size_bytes=self.config.block_size_bytes
                )
                blocks.append(info)
                self.prefix_groups[prefix_id].append(block_id)
            
            if (prefix_id + 1) % 20 == 0:
                print(f"  Generated prefix {prefix_id + 1}/{self.config.num_prefixes}")
        
        return blocks
    
    def _generate_unique_blocks(self, writer: AggregatedWriter, 
                                 target_gb: float, start_idx: int) -> List[BlockInfo]:
        """Generate unique (non-shared) blocks"""
        if target_gb <= 0:
            return []
        
        num_blocks = int(target_gb * 1024**3 / self.config.block_size_bytes)
        print(f"  Target: {target_gb:.2f} GB")
        print(f"  Blocks to generate: {num_blocks:,}")
        
        blocks = []
        report_interval = max(num_blocks // 20, 100)
        
        for i in range(num_blocks):
            global_idx = start_idx + i
            seed = 1000000 + i  # Different seed range from prefixes
            
            # Generate data
            key_data, value_data = self._generate_block_data(seed)
            block_id = self._compute_block_id(key_data, value_data)
            
            # Write to aggregate file
            file_id, offset = writer.write_block(key_data, value_data)
            
            # Create metadata
            info = BlockInfo(
                block_id=block_id,
                block_idx=global_idx,
                is_prefix=False,
                prefix_id=-1,
                token_offset=global_idx * self.config.block_size_tokens,
                agg_file_id=file_id,
                agg_offset=offset,
                size_bytes=self.config.block_size_bytes
            )
            blocks.append(info)
            
            if (i + 1) % report_interval == 0:
                pct = (i + 1) / num_blocks * 100
                print(f"  Progress: {i + 1:,}/{num_blocks:,} ({pct:.1f}%)")
        
        return blocks
    
    def _generate_sessions(self, prefix_blocks: List[BlockInfo], 
                           unique_blocks: List[BlockInfo]):
        """Generate session mappings for workload"""
        blocks_per_prefix = self.config.blocks_per_prefix
        unique_blocks_per_session = 8  # Each session has 8 unique continuation blocks
        
        num_sessions = self.config.num_prefixes * self.config.sessions_per_prefix
        print(f"  Generating {num_sessions} sessions...")
        
        for session_id in range(num_sessions):
            # Assign prefix (round-robin)
            prefix_id = session_id % self.config.num_prefixes
            
            # Get prefix block IDs
            prefix_block_ids = self.prefix_groups[prefix_id]
            
            # Assign unique blocks (cycling through available)
            if unique_blocks:
                unique_start = (session_id * unique_blocks_per_session) % len(unique_blocks)
                unique_block_ids = [
                    unique_blocks[(unique_start + i) % len(unique_blocks)].block_id
                    for i in range(unique_blocks_per_session)
                ]
            else:
                unique_block_ids = []
            
            self.sessions[f"session_{session_id:05d}"] = {
                "prefix_id": prefix_id,
                "prefix_blocks": prefix_block_ids,
                "unique_blocks": unique_block_ids,
                "total_blocks": len(prefix_block_ids) + len(unique_block_ids)
            }
    
    def _save_index(self, output_dir: Path, blocks: List[BlockInfo], num_files: int):
        """Save index files"""
        print("\nSaving index files...")
        
        # Block lookup: block_id -> info
        block_index = {b.block_id: asdict(b) for b in blocks}
        
        # Full index
        index_data = {
            "config": {
                "num_layers": self.config.num_layers,
                "num_kv_heads": self.config.num_kv_heads,
                "head_dim": self.config.head_dim,
                "block_size_tokens": self.config.block_size_tokens,
                "block_size_bytes": self.config.block_size_bytes,
                "blocks_per_aggregate": self.config.blocks_per_aggregate,
            },
            "stats": {
                "total_blocks": len(blocks),
                "prefix_blocks": sum(1 for b in blocks if b.is_prefix),
                "unique_blocks": sum(1 for b in blocks if not b.is_prefix),
                "total_size_gb": len(blocks) * self.config.block_size_bytes / 1024**3,
                "num_aggregate_files": num_files,
                "num_prefixes": self.config.num_prefixes,
                "num_sessions": len(self.sessions),
            },
            "blocks": block_index,
            "prefix_groups": {str(k): v for k, v in self.prefix_groups.items()},
            "sessions": self.sessions,
        }
        
        # Save JSON (human-readable)
        json_path = output_dir / "index.json"
        with open(json_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        print(f"  Saved: {json_path}")
        
        # Save pickle (faster loading)
        pkl_path = output_dir / "index.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"  Saved: {pkl_path}")
        
        # Save lightweight block lookup (for adapters)
        lookup_path = output_dir / "block_lookup.pkl"
        lookup = {
            b.block_id: (b.agg_file_id, b.agg_offset, b.size_bytes)
            for b in blocks
        }
        with open(lookup_path, 'wb') as f:
            pickle.dump(lookup, f)
        print(f"  Saved: {lookup_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate optimized KV cache benchmark data")
    parser.add_argument("--size_gb", type=float, default=100.0, 
                        help="Total data size in GB (default: 100)")
    parser.add_argument("--output", type=str, 
                        default="/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data",
                        help="Output directory")
    parser.add_argument("--block_tokens", type=int, default=256,
                        help="Tokens per block (default: 256)")
    parser.add_argument("--num_prefixes", type=int, default=100,
                        help="Number of unique prefixes (default: 100)")
    parser.add_argument("--sessions_per_prefix", type=int, default=50,
                        help="Sessions per prefix (default: 50)")
    args = parser.parse_args()
    
    config = DataGenConfig(
        output_dir=Path(args.output),
        block_size_tokens=args.block_tokens,
        num_prefixes=args.num_prefixes,
        sessions_per_prefix=args.sessions_per_prefix,
    )
    
    generator = OptimizedDataGenerator(config)
    generator.generate(size_gb=args.size_gb)


if __name__ == "__main__":
    main()
