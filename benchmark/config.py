# benchmark/config.py
"""
Benchmark Configuration for Perlmutter
LLaMA-70B KV Cache Specifications
"""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class LLaMAConfig:
    """LLaMA-70B model dimensions"""
    num_layers: int = 80
    num_heads: int = 64      # GQA: 8 KV heads, but we store full for benchmark
    num_kv_heads: int = 8    # Grouped Query Attention
    head_dim: int = 128
    dtype: str = "float16"
    
    @property
    def kv_size_per_token(self) -> int:
        """Bytes per token for KV cache"""
        # key + value, each: [num_layers, num_kv_heads, head_dim]
        element_size = 2  # float16
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * element_size
    
    # Per token: 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320KB

@dataclass  
class BenchmarkConfig:
    """Benchmark settings"""
    # Paths
    base_path: Path = Path("/pscratch/sd/s/sgkim/Skim-cascade")
    data_path: Path = Path("/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data")
    results_path: Path = Path("/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results")
    
    # Data generation
    total_data_size_gb: float = 500.0
    block_size_tokens: int = 256        # Tokens per block (matches LMCache chunk)
    num_unique_prefixes: int = 100      # Number of unique system prompts
    prefix_length_tokens: int = 2048    # System prompt length
    
    # Workload
    num_sessions: int = 1000            # Concurrent sessions for shared prefix test
    read_write_ratio: float = 0.8       # 80% reads, 20% writes
    
    # Perlmutter specs
    num_gpus_per_node: int = 4
    gpu_memory_gb: float = 40.0
    dram_per_node_gb: float = 256.0
    shm_capacity_gb: float = 128.0
    
    # Systems to benchmark
    systems: tuple = ("cascade", "hdf5", "redis", "lmcache", "pdc")
    
    def __post_init__(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)


# Singleton configs
LLAMA_CONFIG = LLaMAConfig()
BENCHMARK_CONFIG = BenchmarkConfig()
