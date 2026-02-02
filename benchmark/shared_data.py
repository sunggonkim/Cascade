# benchmark/shared_data.py
"""
Shared Data Reader for all benchmark adapters.

All five systems (Cascade, LMCache, HDF5, PDC, Redis) use this
to read the same aggregated KV cache data.

Usage:
    from benchmark.shared_data import SharedDataReader
    
    reader = SharedDataReader("/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data")
    
    # Get a specific block
    block_id, key_data, value_data = reader.get_block_by_id("abc123...")
    
    # Iterate over all blocks
    for block_id, key_data, value_data in reader.iter_blocks():
        adapter.put(block_id, key_data, value_data)
    
    # Get session workload
    for session in reader.iter_sessions():
        for block_id in session['all_blocks']:
            data = reader.get_block_by_id(block_id)
"""
import os
import json
import pickle
import mmap
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass


@dataclass
class BlockLocation:
    """Location of a block in aggregated storage"""
    file_id: int
    offset: int
    size_bytes: int


class SharedDataReader:
    """
    Fast reader for aggregated KV cache data.
    
    Uses memory-mapped files for efficient random access.
    All adapters should use this to ensure they read the same data.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.agg_dir = self.data_dir / "aggregated"
        
        # Load index
        self._load_index()
        
        # Memory-mapped file handles (lazy loading)
        self._mmap_files: Dict[int, mmap.mmap] = {}
        self._file_handles: Dict[int, Any] = {}
    
    def _load_index(self):
        """Load index from pickle (fast) or JSON (fallback)"""
        pkl_path = self.data_dir / "index.pkl"
        json_path = self.data_dir / "index.json"
        lookup_path = self.data_dir / "block_lookup.pkl"
        
        # Load lightweight lookup first
        if lookup_path.exists():
            with open(lookup_path, 'rb') as f:
                self._block_lookup: Dict[str, Tuple[int, int, int]] = pickle.load(f)
            print(f"[SharedDataReader] Loaded block_lookup.pkl ({len(self._block_lookup)} blocks)")
        else:
            self._block_lookup = {}
        
        # Load full index
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                self._index = pickle.load(f)
            print(f"[SharedDataReader] Loaded index.pkl")
        elif json_path.exists():
            with open(json_path, 'r') as f:
                self._index = json.load(f)
            print(f"[SharedDataReader] Loaded index.json")
        else:
            raise FileNotFoundError(f"No index file found in {self.data_dir}")
        
        # Parse config
        self.config = self._index.get('config', {})
        self.stats = self._index.get('stats', {})
        self.block_size = self.config.get('block_size_bytes', 83886080)  # 80MB default
        self.block_tokens = self.config.get('block_size_tokens', 256)
        
        # Block shape for unpacking
        self.num_layers = self.config.get('num_layers', 80)
        self.num_kv_heads = self.config.get('num_kv_heads', 8)
        self.head_dim = self.config.get('head_dim', 128)
        self.block_shape = (self.block_tokens, self.num_layers, self.num_kv_heads, self.head_dim)
        self.single_tensor_bytes = np.prod(self.block_shape) * 2  # float16
        
        # Build lookup if not loaded
        if not self._block_lookup:
            blocks = self._index.get('blocks', {})
            for block_id, info in blocks.items():
                self._block_lookup[block_id] = (
                    info['agg_file_id'],
                    info['agg_offset'],
                    info['size_bytes']
                )
    
    def _get_mmap(self, file_id: int) -> mmap.mmap:
        """Get memory-mapped file handle (lazy loading)"""
        if file_id not in self._mmap_files:
            file_path = self.agg_dir / f"agg_{file_id:06d}.bin"
            if not file_path.exists():
                raise FileNotFoundError(f"Aggregate file not found: {file_path}")
            
            fh = open(file_path, 'rb')
            self._file_handles[file_id] = fh
            self._mmap_files[file_id] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        
        return self._mmap_files[file_id]
    
    def get_block_by_id(self, block_id: str) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Get a block by its content-addressed ID.
        
        Returns:
            (block_id, key_data, value_data) or None if not found
        """
        if block_id not in self._block_lookup:
            return None
        
        file_id, offset, size_bytes = self._block_lookup[block_id]
        mm = self._get_mmap(file_id)
        
        # Read raw bytes
        mm.seek(offset)
        raw_data = mm.read(size_bytes)
        
        # Unpack into key and value
        key_bytes = raw_data[:self.single_tensor_bytes]
        value_bytes = raw_data[self.single_tensor_bytes:]
        
        key_data = np.frombuffer(key_bytes, dtype=np.float16).reshape(self.block_shape)
        value_data = np.frombuffer(value_bytes, dtype=np.float16).reshape(self.block_shape)
        
        return block_id, key_data, value_data
    
    def get_block_raw(self, block_id: str) -> Optional[Tuple[str, bytes, bytes]]:
        """
        Get a block as raw bytes (for adapters that don't need numpy).
        
        Returns:
            (block_id, key_bytes, value_bytes) or None if not found
        """
        if block_id not in self._block_lookup:
            return None
        
        file_id, offset, size_bytes = self._block_lookup[block_id]
        mm = self._get_mmap(file_id)
        
        mm.seek(offset)
        raw_data = mm.read(size_bytes)
        
        key_bytes = raw_data[:self.single_tensor_bytes]
        value_bytes = raw_data[self.single_tensor_bytes:]
        
        return block_id, key_bytes, value_bytes
    
    def iter_blocks(self, limit: int = None) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        """Iterate over all blocks"""
        count = 0
        for block_id in self._block_lookup:
            if limit and count >= limit:
                break
            result = self.get_block_by_id(block_id)
            if result:
                yield result
                count += 1
    
    def iter_blocks_raw(self, limit: int = None) -> Iterator[Tuple[str, bytes, bytes]]:
        """Iterate over all blocks as raw bytes"""
        count = 0
        for block_id in self._block_lookup:
            if limit and count >= limit:
                break
            result = self.get_block_raw(block_id)
            if result:
                yield result
                count += 1
    
    def get_block_ids(self) -> List[str]:
        """Get all block IDs"""
        return list(self._block_lookup.keys())
    
    def get_prefix_block_ids(self) -> List[str]:
        """Get block IDs for prefix blocks only"""
        prefix_groups = self._index.get('prefix_groups', {})
        result = []
        for group_ids in prefix_groups.values():
            result.extend(group_ids)
        return result
    
    def get_prefix_group(self, prefix_id: int) -> List[str]:
        """Get block IDs for a specific prefix group"""
        prefix_groups = self._index.get('prefix_groups', {})
        return prefix_groups.get(str(prefix_id), [])
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session info including block IDs"""
        sessions = self._index.get('sessions', {})
        return sessions.get(session_id)
    
    def iter_sessions(self, limit: int = None) -> Iterator[Dict]:
        """Iterate over sessions"""
        sessions = self._index.get('sessions', {})
        count = 0
        for session_id, session_data in sessions.items():
            if limit and count >= limit:
                break
            yield {'session_id': session_id, **session_data}
            count += 1
    
    @property
    def num_blocks(self) -> int:
        return len(self._block_lookup)
    
    @property
    def num_prefix_blocks(self) -> int:
        return self.stats.get('prefix_blocks', 0)
    
    @property
    def num_sessions(self) -> int:
        return self.stats.get('num_sessions', 0)
    
    @property
    def total_size_gb(self) -> float:
        return self.stats.get('total_size_gb', 0.0)
    
    def close(self):
        """Clean up memory-mapped files"""
        for mm in self._mmap_files.values():
            mm.close()
        for fh in self._file_handles.values():
            fh.close()
        self._mmap_files.clear()
        self._file_handles.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return (f"SharedDataReader(blocks={self.num_blocks}, "
                f"prefixes={self.num_prefix_blocks}, "
                f"sessions={self.num_sessions}, "
                f"size={self.total_size_gb:.2f}GB)")


# =============================================================================
# Helper functions
# =============================================================================

def load_shared_data(data_dir: str = None) -> SharedDataReader:
    """Convenience function to load shared data"""
    if data_dir is None:
        data_dir = "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data"
    return SharedDataReader(data_dir)


def verify_data_integrity(data_dir: str = None, sample_size: int = 100):
    """Verify data integrity by checking content hashes"""
    import hashlib
    
    reader = load_shared_data(data_dir)
    print(f"Verifying {sample_size} blocks from {reader.num_blocks} total...")
    
    errors = 0
    for i, (block_id, key_data, value_data) in enumerate(reader.iter_blocks(limit=sample_size)):
        # Recompute hash
        hasher = hashlib.sha256()
        hasher.update(key_data.tobytes())
        hasher.update(value_data.tobytes())
        computed_hash = hasher.hexdigest()[:32]
        
        if computed_hash != block_id:
            print(f"  ERROR: Block {i} hash mismatch! Expected {block_id}, got {computed_hash}")
            errors += 1
    
    reader.close()
    
    if errors == 0:
        print(f"✓ All {sample_size} blocks verified successfully!")
    else:
        print(f"✗ {errors} blocks failed verification!")
    
    return errors == 0


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Test loading
    reader = load_shared_data(data_dir)
    print(reader)
    print(f"\nConfig: {reader.config}")
    print(f"Stats: {reader.stats}")
    
    # Test reading a few blocks
    print("\nSample blocks:")
    for i, (block_id, key, val) in enumerate(reader.iter_blocks(limit=3)):
        print(f"  Block {i}: id={block_id[:16]}..., key_shape={key.shape}, val_shape={val.shape}")
    
    # Verify integrity
    verify_data_integrity(data_dir, sample_size=10)
    
    reader.close()
