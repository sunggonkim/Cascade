# benchmark/data_generator.py
"""
Generate shared KV cache dataset for all benchmarks.
Creates LLaMA-70B compatible KV blocks.

Usage:
    python -m benchmark.data_generator --size_gb 500 --output /path/to/data
"""
import os
import sys
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from .config import LLAMA_CONFIG, BENCHMARK_CONFIG


@dataclass
class BlockMetadata:
    """Metadata for a generated KV block"""
    block_id: str
    block_index: int
    num_tokens: int
    size_bytes: int
    is_prefix: bool
    prefix_id: int          # Which prefix group (0-99 for 100 prefixes)
    token_offset: int       # Starting token position
    content_hash: str       # SHA-256 of content (for dedup verification)


class KVDataGenerator:
    """
    Generate realistic KV cache data for benchmarking.
    
    Data layout:
    - 100 unique system prompts (prefixes), each 2048 tokens
    - Each prompt shared by ~10 sessions
    - Remaining tokens are unique per session
    
    This creates realistic deduplication opportunities.
    """
    
    def __init__(self, config: BenchmarkConfig = BENCHMARK_CONFIG):
        self.config = config
        self.llama = LLAMA_CONFIG
        self.metadata: List[BlockMetadata] = []
        
        # Calculate block dimensions
        self.block_tokens = config.block_size_tokens
        self.block_shape = (
            self.block_tokens,
            self.llama.num_layers,
            self.llama.num_kv_heads,
            self.llama.head_dim
        )
        self.block_size_bytes = np.prod(self.block_shape) * 2 * 2  # key + value, float16
        
        print(f"Block shape: {self.block_shape}")
        print(f"Block size: {self.block_size_bytes / 1024 / 1024:.2f} MB")
    
    def generate_block(self, block_index: int, is_prefix: bool, prefix_id: int, 
                       seed: int) -> Tuple[np.ndarray, np.ndarray, BlockMetadata]:
        """Generate a single KV block with deterministic content."""
        rng = np.random.RandomState(seed)
        
        # Generate KV data (realistic distribution)
        # Real KV caches have values roughly normally distributed
        key_data = rng.randn(*self.block_shape).astype(np.float16) * 0.1
        value_data = rng.randn(*self.block_shape).astype(np.float16) * 0.1
        
        # Compute content hash for dedup verification
        content = key_data.tobytes() + value_data.tobytes()
        content_hash = hashlib.sha256(content).hexdigest()[:32]
        
        # Block ID is content-addressed (like Cascade)
        block_id = content_hash
        
        metadata = BlockMetadata(
            block_id=block_id,
            block_index=block_index,
            num_tokens=self.block_tokens,
            size_bytes=self.block_size_bytes,
            is_prefix=is_prefix,
            prefix_id=prefix_id,
            token_offset=block_index * self.block_tokens,
            content_hash=content_hash
        )
        
        return key_data, value_data, metadata
    
    def generate_prefix_blocks(self, output_dir: Path) -> List[BlockMetadata]:
        """Generate shared prefix blocks (system prompts)."""
        prefix_dir = output_dir / "prefixes"
        prefix_dir.mkdir(exist_ok=True)
        
        prefix_metadata = []
        blocks_per_prefix = self.config.prefix_length_tokens // self.block_tokens
        
        print(f"\\nGenerating {self.config.num_unique_prefixes} prefixes...")
        print(f"  Blocks per prefix: {blocks_per_prefix}")
        
        for prefix_id in range(self.config.num_unique_prefixes):
            for block_idx in range(blocks_per_prefix):
                global_idx = prefix_id * blocks_per_prefix + block_idx
                seed = prefix_id * 10000 + block_idx  # Deterministic
                
                key_data, value_data, meta = self.generate_block(
                    block_index=global_idx,
                    is_prefix=True,
                    prefix_id=prefix_id,
                    seed=seed
                )
                
                # Save block
                block_path = prefix_dir / f"prefix_{prefix_id:03d}_block_{block_idx:03d}.npz"
                np.savez_compressed(block_path, key=key_data, value=value_data)
                
                prefix_metadata.append(meta)
            
            if (prefix_id + 1) % 10 == 0:
                print(f"  Generated prefix {prefix_id + 1}/{self.config.num_unique_prefixes}")
        
        return prefix_metadata
    
    def generate_unique_blocks(self, output_dir: Path, target_size_gb: float,
                               prefix_metadata: List[BlockMetadata]) -> List[BlockMetadata]:
        """Generate unique (non-shared) blocks to fill remaining space."""
        unique_dir = output_dir / "unique"
        unique_dir.mkdir(exist_ok=True)
        
        prefix_size_bytes = sum(m.size_bytes for m in prefix_metadata)
        target_bytes = int(target_size_gb * 1024**3) - prefix_size_bytes
        num_unique_blocks = target_bytes // self.block_size_bytes
        
        print(f"\\nGenerating {num_unique_blocks:,} unique blocks...")
        print(f"  Target size: {target_bytes / 1024**3:.2f} GB")
        
        unique_metadata = []
        start_time = time.time()
        
        for i in range(num_unique_blocks):
            seed = 1000000 + i  # Different seed range from prefixes
            
            key_data, value_data, meta = self.generate_block(
                block_index=len(prefix_metadata) + i,
                is_prefix=False,
                prefix_id=-1,
                seed=seed
            )
            
            # Save in sharded directories (avoid too many files per dir)
            shard = i // 1000
            shard_dir = unique_dir / f"shard_{shard:05d}"
            shard_dir.mkdir(exist_ok=True)
            
            block_path = shard_dir / f"block_{i:08d}.npz"
            np.savez_compressed(block_path, key=key_data, value=value_data)
            
            unique_metadata.append(meta)
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_unique_blocks - i - 1) / rate
                print(f"  Generated {i + 1:,}/{num_unique_blocks:,} blocks "
                      f"({rate:.1f} blocks/s, ETA: {eta/60:.1f} min)")
        
        return unique_metadata
    
    def generate_session_mappings(self, output_dir: Path, 
                                   prefix_metadata: List[BlockMetadata],
                                   unique_metadata: List[BlockMetadata]) -> Dict:
        """
        Generate session-to-block mappings for shared prefix workload.
        
        Each session:
        - Uses one of 100 prefixes (shared)
        - Has unique continuation blocks
        """
        sessions = {}
        blocks_per_prefix = self.config.prefix_length_tokens // self.block_tokens
        unique_blocks_per_session = 10  # Each session has 10 unique blocks after prefix
        
        for session_id in range(self.config.num_sessions):
            # Assign to a prefix (round-robin for even distribution)
            prefix_id = session_id % self.config.num_unique_prefixes
            
            # Get prefix blocks
            prefix_start = prefix_id * blocks_per_prefix
            prefix_block_ids = [
                prefix_metadata[prefix_start + i].block_id 
                for i in range(blocks_per_prefix)
            ]
            
            # Assign unique blocks
            unique_start = (session_id * unique_blocks_per_session) % len(unique_metadata)
            unique_block_ids = [
                unique_metadata[(unique_start + i) % len(unique_metadata)].block_id
                for i in range(unique_blocks_per_session)
            ]
            
            sessions[f"session_{session_id:05d}"] = {
                "prefix_id": prefix_id,
                "prefix_blocks": prefix_block_ids,
                "unique_blocks": unique_block_ids,
                "all_blocks": prefix_block_ids + unique_block_ids
            }
        
        # Save mappings
        mapping_path = output_dir / "session_mappings.json"
        with open(mapping_path, 'w') as f:
            json.dump(sessions, f, indent=2)
        
        return sessions
    
    def generate_all(self, output_dir: Path = None, size_gb: float = None):
        """Generate complete dataset."""
        output_dir = output_dir or self.config.data_path
        size_gb = size_gb or self.config.total_data_size_gb
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("KV Cache Benchmark Data Generator")
        print("=" * 60)
        print(f"Output: {output_dir}")
        print(f"Target size: {size_gb} GB")
        print(f"Model: LLaMA-70B")
        print(f"Block size: {self.block_tokens} tokens = {self.block_size_bytes/1024/1024:.2f} MB")
        
        start_time = time.time()
        
        # 1. Generate prefix blocks
        prefix_metadata = self.generate_prefix_blocks(output_dir)
        
        # 2. Generate unique blocks
        unique_metadata = self.generate_unique_blocks(output_dir, size_gb, prefix_metadata)
        
        # 3. Generate session mappings
        all_metadata = prefix_metadata + unique_metadata
        sessions = self.generate_session_mappings(output_dir, prefix_metadata, unique_metadata)
        
        # 4. Save metadata index
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "config": {
                    "llama": asdict(self.llama),
                    "block_tokens": self.block_tokens,
                    "block_size_bytes": self.block_size_bytes,
                    "num_prefixes": self.config.num_unique_prefixes,
                    "num_sessions": self.config.num_sessions,
                },
                "blocks": [asdict(m) for m in all_metadata],
                "stats": {
                    "total_blocks": len(all_metadata),
                    "prefix_blocks": len(prefix_metadata),
                    "unique_blocks": len(unique_metadata),
                    "total_size_gb": sum(m.size_bytes for m in all_metadata) / 1024**3,
                }
            }, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"\\n{'=' * 60}")
        print(f"Generation complete in {elapsed/60:.1f} minutes")
        print(f"Total blocks: {len(all_metadata):,}")
        print(f"Prefix blocks: {len(prefix_metadata):,} ({len(prefix_metadata)*100/len(all_metadata):.1f}%)")
        print(f"Sessions: {len(sessions):,}")
        print(f"Output: {output_dir}")
        print("=" * 60)
        
        return all_metadata, sessions


def main():
    parser = argparse.ArgumentParser(description="Generate KV cache benchmark data")
    parser.add_argument("--size_gb", type=float, default=500.0, help="Total data size in GB")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--block_tokens", type=int, default=256, help="Tokens per block")
    args = parser.parse_args()
    
    config = BENCHMARK_CONFIG
    if args.block_tokens:
        config.block_size_tokens = args.block_tokens
    
    generator = KVDataGenerator(config)
    generator.generate_all(
        output_dir=Path(args.output) if args.output else None,
        size_gb=args.size_gb
    )


if __name__ == "__main__":
    main()
