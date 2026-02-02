# benchmark/adapters/base.py
"""
Abstract base class for storage system adapters.
All systems must implement this interface for fair comparison.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import time


@dataclass
class BenchmarkStats:
    """Statistics collected during benchmark"""
    system_name: str
    operation: str  # "write", "read", "mixed"
    
    # Throughput
    total_ops: int = 0
    total_bytes: int = 0
    duration_seconds: float = 0.0
    
    # Latency (in milliseconds)
    latencies: List[float] = field(default_factory=list)
    
    # Cache stats
    hits: int = 0
    misses: int = 0
    
    # Errors
    errors: int = 0
    
    @property
    def ops_per_second(self) -> float:
        return self.total_ops / max(self.duration_seconds, 0.001)
    
    @property
    def throughput_gbps(self) -> float:
        return (self.total_bytes / 1024**3) / max(self.duration_seconds, 0.001)
    
    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)
    
    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[len(sorted_lat) // 2]
    
    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[int(len(sorted_lat) * 0.99)]
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)
    
    def to_dict(self) -> Dict:
        return {
            "system": self.system_name,
            "operation": self.operation,
            "total_ops": self.total_ops,
            "total_bytes": self.total_bytes,
            "duration_s": self.duration_seconds,
            "ops_per_second": self.ops_per_second,
            "throughput_gbps": self.throughput_gbps,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "errors": self.errors,
        }


class StorageAdapter(ABC):
    """
    Abstract base class for KV cache storage systems.
    
    All adapters must implement:
    - put(): Store a block
    - get(): Retrieve a block
    - contains(): Check existence
    - clear(): Reset state
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the storage system. Returns True if successful."""
        pass
    
    @abstractmethod
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        """
        Store a KV block.
        
        Args:
            block_id: Unique identifier (content-addressed hash)
            key_data: Key tensor bytes
            value_data: Value tensor bytes
        
        Returns:
            True if stored successfully
        """
        pass
    
    @abstractmethod
    def get(self, block_id: str) -> Optional[tuple]:
        """
        Retrieve a KV block.
        
        Args:
            block_id: Block identifier
        
        Returns:
            Tuple of (key_data, value_data) if found, None otherwise
        """
        pass
    
    @abstractmethod
    def contains(self, block_id: str) -> bool:
        """Check if block exists."""
        pass
    
    @abstractmethod
    def delete(self, block_id: str) -> bool:
        """Delete a block."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered writes."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get system-specific statistics."""
        pass
    
    def is_available(self) -> bool:
        """Check if this storage system is available."""
        return self._initialized
    
    def close(self) -> None:
        """Cleanup resources."""
        pass
