# 🛠️ Development Guide

This document provides detailed instructions for developers working on the Cascade codebase.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Organization](#code-organization)
3. [Core Components](#core-components)
4. [Building from Source](#building-from-source)
5. [Adding Features](#adding-features)
6. [Testing](#testing)
7. [Performance Profiling](#performance-profiling)
8. [Coding Conventions](#coding-conventions)

---

## Architecture Overview

Cascade is a 4-tier hierarchical KV cache storage system:

```
┌─────────────────────────────────────────────────────────────┐
│                        Application                           │
│                    (vLLM, LMCache, etc.)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      CascadeStore                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ DedupIndex  │  │ TierManager  │  │ EvictionPolicy    │  │
│  │ (SHA-256)   │  │ (GPU→SHM→L) │  │ (Semantic LRU)    │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │                   │                    │
          ▼                   ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
│  GPU Backend │  │  SHM Backend │  │   Lustre Backend     │
│  (CUDA API)  │  │  (mmap)      │  │   (Aggregated I/O)   │
└──────────────┘  └──────────────┘  └──────────────────────┘
```

---

## Code Organization

### C++ Implementation (`cascade_Code/cpp/`)

```
cpp/
├── include/
│   └── cascade.hpp        # Main header (all declarations)
├── src/
│   ├── cascade_core.cpp   # Core logic
│   │   ├── ShardedIndex   # Concurrent hash map
│   │   ├── ShmBackend     # mmap-based shared memory
│   │   ├── LustreBackend  # io_uring async file I/O
│   │   └── CascadeStore   # 3-tier orchestration
│   ├── gpu_backend.cu     # CUDA GPU tier
│   └── benchmark.cpp      # Standalone benchmark
├── python/
│   └── bindings.cpp       # pybind11 Python bindings
├── CMakeLists.txt
└── build_perlmutter.sh    # Build script for NERSC
```

### Python Adapters (`benchmark/adapters/`)

```
adapters/
├── base.py                # Abstract base class
├── cascade_adapter.py     # Cascade Python wrapper
├── lmcache_adapter.py     # LMCache adapter
├── hdf5_adapter.py        # HDF5 adapter
├── redis_adapter.py       # Redis adapter
└── pdc_adapter.py         # PDC adapter
```

---

## Core Components

### 1. Block ID Computation (SHA-256)

All block IDs are content-addressed using SHA-256:

```cpp
// C++ (cascade_core.cpp)
BlockId compute_block_id(const uint8_t* data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data, size, hash);
    
    char hex[33];
    for (int i = 0; i < 16; i++) {
        snprintf(hex + i*2, 3, "%02x", hash[i]);
    }
    hex[32] = '\0';
    return std::string(hex);
}
```

```python
# Python equivalent
import hashlib

def compute_block_id(key: bytes, value: bytes) -> str:
    h = hashlib.sha256()
    h.update(key)
    h.update(value)
    return h.hexdigest()[:32]
```

### 2. ShardedIndex (Concurrent Hash Map)

Lock-free-like design with per-shard mutexes:

```cpp
template<typename V>
class ShardedIndex {
    static constexpr size_t NUM_SHARDS = 256;
    
    struct Shard {
        std::shared_mutex mutex;
        std::unordered_map<BlockId, V> data;
        std::unordered_map<BlockId, size_t> sizes;
        size_t total_size = 0;
    };
    
    std::array<Shard, NUM_SHARDS> shards_;
    
    size_t get_shard_id(const BlockId& key) {
        return std::hash<std::string>{}(key) % NUM_SHARDS;
    }
};
```

### 3. GPU Backend (CUDA)

```cpp
// gpu_backend.cu
class GPUBackend {
    void* gpu_buffer_;
    size_t capacity_;
    ShardedIndex<size_t> index_;  // block_id → offset
    
public:
    bool put(const BlockId& id, const void* data, size_t size) {
        // Allocate from GPU memory pool
        size_t offset = allocate(size);
        if (offset == SIZE_MAX) return false;
        
        // Async copy H2D
        cudaMemcpyAsync(gpu_buffer_ + offset, data, size, 
                        cudaMemcpyHostToDevice, stream_);
        
        index_.put(id, offset, size);
        return true;
    }
};
```

### 4. SHM Backend (mmap)

```cpp
// cascade_core.cpp
class ShmBackend {
    void* shm_base_;
    int shm_fd_;
    
public:
    ShmBackend(const std::string& name, size_t capacity) {
        shm_fd_ = shm_open(name.c_str(), O_RDWR | O_CREAT, 0666);
        ftruncate(shm_fd_, capacity);
        shm_base_ = mmap(nullptr, capacity, PROT_READ | PROT_WRITE,
                         MAP_SHARED, shm_fd_, 0);
    }
    
    bool put(const BlockId& id, const void* data, size_t size) {
        size_t offset = allocate(size);
        memcpy(static_cast<char*>(shm_base_) + offset, data, size);
        return true;
    }
};
```

### 5. Lustre Backend (Aggregated I/O)

```cpp
// cascade_core.cpp
class LustreBackend {
    std::filesystem::path lustre_path_;
    std::ofstream current_file_;
    size_t current_offset_ = 0;
    
public:
    LustreBackend(const std::filesystem::path& path) : lustre_path_(path) {
        // Set Lustre stripe configuration
        std::string cmd = "lfs setstripe -c 16 -S 4m " + path.string();
        std::system(cmd.c_str());
    }
    
    bool put(const BlockId& id, const void* data, size_t size) {
        // Write to aggregated file (not one file per block!)
        current_file_.write(static_cast<const char*>(data), size);
        index_.put(id, {current_file_id_, current_offset_}, size);
        current_offset_ += size;
        return true;
    }
};
```

---

## Building from Source

### Prerequisites

```bash
module load python cudatoolkit cray-mpich cmake
```

### Build C++ Library

```bash
cd cascade_Code/cpp

# Create build directory
mkdir -p build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DPYTHON_EXECUTABLE=$(which python3)

# Build
make -j$(nproc)

# Install Python bindings
cp cascade_cpp*.so ../python/
```

### Build Script

`build_perlmutter.sh`:
```bash
#!/bin/bash
module load python cudatoolkit cray-mpich cmake

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

---

## Adding Features

### Adding a New Storage Tier

1. Create backend class in `cascade_core.cpp`:

```cpp
class NewTierBackend {
public:
    NewTierBackend(const Config& config);
    bool put(const BlockId& id, const void* data, size_t size);
    std::vector<uint8_t> get(const BlockId& id);
    bool contains(const BlockId& id);
    void evict(const BlockId& id);
};
```

2. Integrate with `CascadeStore`:

```cpp
class CascadeStore {
    // Existing tiers
    GPUBackend gpu_;
    ShmBackend shm_;
    LustreBackend lustre_;
    
    // Add new tier
    NewTierBackend new_tier_;
    
    bool put(const BlockId& id, const void* data, size_t size) {
        // Try new tier in the hierarchy
        if (gpu_.put(...)) return true;
        if (new_tier_.put(...)) return true;  // New
        if (shm_.put(...)) return true;
        return lustre_.put(...);
    }
};
```

3. Update Python bindings in `python/bindings.cpp`

### Adding a Benchmark Adapter

1. Create adapter file:

```python
# benchmark/adapters/new_adapter.py
from .base import StorageAdapter

class NewAdapter(StorageAdapter):
    def __init__(self, config):
        super().__init__("NewSystem", config)
        
    def initialize(self) -> bool:
        # Setup your system
        return True
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        # Store operation
        return True
    
    def get(self, block_id: str) -> Optional[tuple]:
        # Retrieve operation
        return (key_data, value_data)
```

2. Register in `benchmark/adapters/__init__.py`:

```python
from .new_adapter import NewAdapter
ADAPTERS['newsystem'] = NewAdapter
```

---

## Testing

### Unit Tests

```bash
cd cascade_Code/cpp/build
ctest --output-on-failure
```

### Integration Tests

```bash
# Test Python bindings
python -c "
import cascade_cpp
store = cascade_cpp.CascadeStore()
print('OK')
"
```

### Benchmark Smoke Test

```bash
# Quick test on login node (no GPU)
python benchmark/run_benchmark.py --system cascade --workload smoke_test
```

---

## Performance Profiling

### NVIDIA Nsight

```bash
srun nsys profile -o cascade_profile python benchmark/run_benchmark.py
nsys-ui cascade_profile.qdrep
```

### Python Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run benchmark
run_benchmark()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

### I/O Profiling (Lustre)

```bash
# Use Lustre client stats
lctl get_param llite.*.stats

# Or use Darshan
module load darshan
srun ... python benchmark/run_benchmark.py
darshan-parser $DARSHAN_LOG/*.darshan
```

---

## Coding Conventions

### C++ Style

- Use snake_case for functions and variables
- Use PascalCase for classes and types
- Use SCREAMING_CASE for constants
- 4-space indentation
- Braces on same line for functions

```cpp
class MyClass {
public:
    void do_something(int param) {
        const int MAX_VALUE = 100;
        if (param > MAX_VALUE) {
            handle_error();
        }
    }
};
```

### Python Style

- Follow PEP 8
- Use type hints
- Use dataclasses for config

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    gpu_capacity_gb: float = 32.0
    shm_capacity_gb: float = 64.0
    
def process_block(block_id: str, data: bytes) -> bool:
    """Process a single KV cache block."""
    return True
```

### Git Commit Messages

```
<type>: <short description>

<longer description if needed>

Types:
- feat: New feature
- fix: Bug fix
- perf: Performance improvement
- docs: Documentation
- refactor: Code refactoring
- test: Tests
- build: Build system
```

Example:
```
feat: Add NVMe tier between SHM and Lustre

Implements a new NVMe storage tier using io_uring for
async I/O. Configured via nvme_path in CascadeConfig.

Benchmarks show 2.3x improvement over direct Lustre writes.
```

---

## Contact

For development questions:
- **Author**: Sung Gon Kim
- **Email**: sgkim@lbl.gov
