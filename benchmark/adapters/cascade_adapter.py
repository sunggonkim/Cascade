# benchmark/adapters/cascade_adapter.py
"""
Adapter for Cascade KV Cache System - Updated for latest API.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add cascade to path
sys.path.insert(0, str(Path("/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src")))

try:
    from .base import StorageAdapter
except ImportError:
    from base import StorageAdapter


class CascadeAdapter(StorageAdapter):
    """
    Adapter for Cascade tiered KV cache.
    
    Features tested:
    - Content-addressed deduplication
    - 3-tier storage (GPU → SHM → Lustre)
    - Semantic eviction with LRU
    - INT4 KV compression
    - ShardedIndex for lock-free-like concurrency
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("Cascade", config)
        self.store = None
        
        # Default config
        self.gpu_capacity_gb = config.get("gpu_capacity_gb", 32.0)
        self.shm_capacity_gb = config.get("shm_capacity_gb", 64.0)
        self.lustre_path = config.get("lustre_path", "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/cascade_store")
        self.use_gpu = config.get("use_gpu", False)
        self.use_compression = config.get("use_compression", True)
        self.use_sharding = config.get("use_sharding", True)
    
    def initialize(self) -> bool:
        try:
            # Updated import - use cascade package
            from cascade import CascadeStore, CascadeConfig, create_login_node_store
            
            if self.use_gpu:
                cfg = CascadeConfig(
                    gpu_capacity_gb=self.gpu_capacity_gb,
                    shm_capacity_gb=self.shm_capacity_gb,
                    lustre_path=self.lustre_path,
                    dedup_enabled=True,
                    prefix_aware=True,
                )
                self.store = CascadeStore(cfg)
            else:
                # Use login node store for testing without GPU
                self.store = create_login_node_store()
            
            self._initialized = True
            print(f"[CascadeAdapter] Initialized (GPU={self.use_gpu})")
            return True
            
        except ImportError as e:
            print(f"[CascadeAdapter] Import error: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"[CascadeAdapter] Init error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized:
            return False
        
        # Combine key and value for storage
        data = key_data + value_data
        return self.store.put(block_id, data, is_prefix=False)
    
    def put_prefix(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        """Store a prefix block (for dedup testing)."""
        if not self._initialized:
            return False
        
        data = key_data + value_data
        return self.store.put(block_id, data, is_prefix=True)
    
    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized:
            return None
        
        data = self.store.get(block_id)
        if data is None:
            return None
        
        # Split back into key and value
        mid = len(data) // 2
        return (data[:mid], data[mid:])
    
    def contains(self, block_id: str) -> bool:
        if not self._initialized:
            return False
        return self.store.contains(block_id)
    
    def delete(self, block_id: str) -> bool:
        # Cascade uses automatic eviction, not explicit delete
        return False
    
    def clear(self) -> None:
        if self._initialized and self.store:
            self.store.clear_cache()
    
    def flush(self) -> None:
        if self._initialized and self.store:
            self.store.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        if not self._initialized:
            return {}
        return self.store.get_stats()
    
    def close(self) -> None:
        if self.store:
            try:
                self.store.cleanup()
            except:
                self.store.flush()
        self._initialized = False
