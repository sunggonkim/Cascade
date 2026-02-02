# benchmark/adapters/hdf5_adapter.py
"""
Adapter for HDF5 storage (baseline comparison).
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter


class HDF5Adapter(StorageAdapter):
    """
    HDF5-based KV cache storage.
    
    Uses h5py with chunked storage and compression.
    This is a common baseline for scientific data storage.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("HDF5", config)
        self.h5file = None
        self.file_path = config.get("file_path", "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/hdf5_store/kv_cache.h5")
        self.compression = config.get("compression", "gzip")
        self.compression_level = config.get("compression_level", 1)  # Fast compression
        
        # Stats
        self._reads = 0
        self._writes = 0
    
    def initialize(self) -> bool:
        try:
            import h5py
            
            # Create directory
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Open/create HDF5 file
            self.h5file = h5py.File(self.file_path, 'a')
            
            # Create groups if needed
            if 'keys' not in self.h5file:
                self.h5file.create_group('keys')
            if 'values' not in self.h5file:
                self.h5file.create_group('values')
            
            self._initialized = True
            return True
            
        except ImportError:
            print("[HDF5Adapter] h5py not installed")
            return False
        except Exception as e:
            print(f"[HDF5Adapter] Init error: {e}")
            return False
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized:
            return False
        
        try:
            # Convert to numpy for HDF5
            key_arr = np.frombuffer(key_data, dtype=np.float16)
            val_arr = np.frombuffer(value_data, dtype=np.float16)
            
            # Store with compression
            if block_id in self.h5file['keys']:
                del self.h5file['keys'][block_id]
                del self.h5file['values'][block_id]
            
            self.h5file['keys'].create_dataset(
                block_id, data=key_arr,
                compression=self.compression,
                compression_opts=self.compression_level
            )
            self.h5file['values'].create_dataset(
                block_id, data=val_arr,
                compression=self.compression,
                compression_opts=self.compression_level
            )
            
            self._writes += 1
            return True
            
        except Exception as e:
            print(f"[HDF5Adapter] Put error: {e}")
            return False
    
    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized:
            return None
        
        try:
            if block_id not in self.h5file['keys']:
                return None
            
            key_arr = self.h5file['keys'][block_id][:]
            val_arr = self.h5file['values'][block_id][:]
            
            self._reads += 1
            return (key_arr.tobytes(), val_arr.tobytes())
            
        except Exception as e:
            return None
    
    def contains(self, block_id: str) -> bool:
        if not self._initialized:
            return False
        return block_id in self.h5file['keys']
    
    def delete(self, block_id: str) -> bool:
        if not self._initialized:
            return False
        
        try:
            if block_id in self.h5file['keys']:
                del self.h5file['keys'][block_id]
                del self.h5file['values'][block_id]
                return True
            return False
        except:
            return False
    
    def clear(self) -> None:
        if self._initialized and self.h5file:
            # Delete and recreate groups
            del self.h5file['keys']
            del self.h5file['values']
            self.h5file.create_group('keys')
            self.h5file.create_group('values')
            self._reads = 0
            self._writes = 0
    
    def flush(self) -> None:
        if self._initialized and self.h5file:
            self.h5file.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "reads": self._reads,
            "writes": self._writes,
            "file_size_mb": os.path.getsize(self.file_path) / 1024**2 if os.path.exists(self.file_path) else 0
        }
    
    def close(self) -> None:
        if self.h5file:
            self.h5file.close()
            self.h5file = None
