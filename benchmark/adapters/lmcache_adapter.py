# benchmark/adapters/lmcache_adapter.py
"""
Adapter for LMCache.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter


class LMCacheAdapter(StorageAdapter):
    """
    Adapter for LMCache KV cache system.
    
    Tests LMCache's disk offloading capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LMCache", config)
        self.cache = None
        self.storage_path = config.get("storage_path", "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/lmcache_store")
        self.max_size_gb = config.get("max_size_gb", 100.0)
        
        self._fallback_mode = False
        self._fallback_store = {}
    
    def initialize(self) -> bool:
        try:
            # Try to import LMCache
            lmcache_path = Path("/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache")
            if lmcache_path.exists():
                sys.path.insert(0, str(lmcache_path))
            
            from lmcache.storage_backend import LocalDiskBackend
            
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            
            self.cache = LocalDiskBackend(
                path=self.storage_path,
                max_size=int(self.max_size_gb * 1024**3)
            )
            
            self._initialized = True
            return True
            
        except ImportError as e:
            print(f"[LMCacheAdapter] LMCache not available: {e}")
            print("[LMCacheAdapter] Using fallback dict-based storage")
            self._fallback_mode = True
            self._initialized = True
            return True
        except Exception as e:
            print(f"[LMCacheAdapter] Init error: {e}")
            self._fallback_mode = True
            self._initialized = True
            return True
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized:
            return False
        
        if self._fallback_mode:
            self._fallback_store[block_id] = (key_data, value_data)
            return True
        
        try:
            data = key_data + value_data
            self.cache.put(block_id, data)
            return True
        except Exception as e:
            return False
    
    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized:
            return None
        
        if self._fallback_mode:
            return self._fallback_store.get(block_id)
        
        try:
            data = self.cache.get(block_id)
            if data is None:
                return None
            mid = len(data) // 2
            return (data[:mid], data[mid:])
        except:
            return None
    
    def contains(self, block_id: str) -> bool:
        if self._fallback_mode:
            return block_id in self._fallback_store
        
        try:
            return self.cache.contains(block_id)
        except:
            return False
    
    def delete(self, block_id: str) -> bool:
        if self._fallback_mode:
            if block_id in self._fallback_store:
                del self._fallback_store[block_id]
                return True
            return False
        
        try:
            return self.cache.delete(block_id)
        except:
            return False
    
    def clear(self) -> None:
        if self._fallback_mode:
            self._fallback_store.clear()
        elif self.cache:
            try:
                self.cache.clear()
            except:
                pass
    
    def flush(self) -> None:
        if not self._fallback_mode and self.cache:
            try:
                self.cache.flush()
            except:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        if self._fallback_mode:
            return {
                "mode": "fallback",
                "num_blocks": len(self._fallback_store),
                "size_bytes": sum(len(k) + len(v) for k, v in self._fallback_store.values())
            }
        
        try:
            return self.cache.get_stats()
        except:
            return {}
    
    def is_available(self) -> bool:
        return self._initialized and not self._fallback_mode
