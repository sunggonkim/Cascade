/**
 * Cascade KV Cache - High-Performance C++ Core
 * 
 * Zero-copy GPU transfers, lock-free indexing, io_uring async I/O
 * 
 * Target: 25+ GB/s on PCIe Gen4, near-hardware limits
 * 
 * Author: SC26 Cascade Team
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <functional>
#include <optional>
#include <cstdint>
#include <cstring>

namespace cascade {

// ============================================================================
// Configuration
// ============================================================================

struct CascadeConfig {
    // GPU tier
    size_t gpu_capacity_bytes = 32ULL * 1024 * 1024 * 1024;  // 32GB
    int gpu_device_id = 0;
    bool use_gpu = true;
    
    // SHM tier (DRAM)
    size_t shm_capacity_bytes = 64ULL * 1024 * 1024 * 1024;  // 64GB
    std::string shm_path = "/dev/shm/cascade";
    
    // Lustre tier
    std::string lustre_path = "/pscratch/sd/s/sgkim/cascade_store";
    size_t lustre_stripe_size = 1024 * 1024;  // 1MB
    int lustre_stripe_count = 16;
    
    // Features
    bool dedup_enabled = true;
    bool compression_enabled = true;
    int num_io_threads = 8;
};

// ============================================================================
// Block ID (32-byte SHA256 prefix as string)
// ============================================================================

using BlockId = std::string;

BlockId compute_block_id(const uint8_t* data, size_t size);

// ============================================================================
// Lock-Free Sharded Index (256 shards, minimal contention)
// ============================================================================

template<typename V>
class ShardedIndex {
public:
    static constexpr size_t NUM_SHARDS = 256;
    
    ShardedIndex();
    ~ShardedIndex();
    
    bool put(const BlockId& key, V value, size_t size = 0);
    std::optional<V> get(const BlockId& key);
    bool remove(const BlockId& key);
    bool contains(const BlockId& key) const;
    
    size_t total_size() const;
    size_t total_count() const;
    void clear();
    
private:
    struct Shard {
        mutable std::shared_mutex mutex;
        std::unordered_map<BlockId, V> data;
        std::unordered_map<BlockId, size_t> sizes;
        std::atomic<size_t> total_size{0};
    };
    
    std::array<Shard, NUM_SHARDS> shards_;
    
    size_t get_shard_id(const BlockId& key) const {
        return std::hash<BlockId>{}(key) % NUM_SHARDS;
    }
};

// ============================================================================
// GPU Backend (CUDA + Memory Pool + Multi-Stream)
// ============================================================================

class GPUMemoryPool;  // Forward declaration

class GPUBackend {
public:
    GPUBackend(size_t capacity_bytes, int device_id = 0);
    ~GPUBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    
    void clear();
    
private:
    size_t capacity_;
    int device_id_;
    std::atomic<size_t> used_{0};
    
    // Memory pool (pre-allocated GPU memory)
    std::unique_ptr<GPUMemoryPool> memory_pool_;
    
    // 32 pinned buffers for parallel transfers (one per thread)
    static constexpr int NUM_PINNED_BUFFERS = 32;
    void* pinned_buffers_[32] = {nullptr};
    void* pinned_buffer_ = nullptr;  // Legacy
    size_t pinned_size_ = 64 * 1024 * 1024;  // 64MB per staging buffer
    
    // 8 CUDA streams for maximum concurrency
    static constexpr int NUM_STREAMS = 8;
    void* cuda_streams_[8] = {nullptr};
    std::atomic<int> current_stream_{0};
    
    // Index: block_id -> (gpu_ptr, size)
    struct GPUBlock {
        void* ptr;
        size_t size;
    };
    ShardedIndex<GPUBlock> index_;
    
    bool init_cuda();
    void* alloc_gpu(size_t size);
    void free_gpu(void* ptr);
    void copy_h2d_async(void* dst, const void* src, size_t size, int stream);
    void copy_d2h_async(void* dst, const void* src, size_t size, int stream);
    void sync_stream(int stream);
};

// ============================================================================
// SHM Backend (mmap + Lock-Free)
// ============================================================================

class ShmBackend {
public:
    ShmBackend(size_t capacity_bytes, const std::string& path = "/dev/shm/cascade");
    ~ShmBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    
    void clear();
    
private:
    size_t capacity_;
    std::string path_;
    std::atomic<size_t> used_{0};
    
    // Memory-mapped region
    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::atomic<size_t> write_offset_{0};
    
    // Index
    struct ShmBlock {
        size_t offset;
        size_t size;
    };
    ShardedIndex<ShmBlock> index_;
};

// ============================================================================
// Lustre Backend (io_uring async I/O)
// ============================================================================

class LustreBackend {
public:
    LustreBackend(const std::string& path, size_t stripe_size = 1024*1024, int stripe_count = 16);
    ~LustreBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    
    void flush();
    
private:
    std::string base_path_;
    size_t stripe_size_;
    int stripe_count_;
    
    // io_uring for async I/O
    void* ring_ = nullptr;
    
    std::string block_path(const BlockId& id) const;
};

// ============================================================================
// CascadeStore - Main 3-Tier Store
// ============================================================================

class CascadeStore {
public:
    explicit CascadeStore(const CascadeConfig& config);
    ~CascadeStore();
    
    // Main API
    bool put(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix = false);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool contains(const BlockId& id) const;
    
    // Batch API (higher throughput)
    size_t put_batch(const std::vector<BlockId>& ids, 
                     const std::vector<const uint8_t*>& data,
                     const std::vector<size_t>& sizes);
    size_t get_batch(const std::vector<BlockId>& ids,
                     std::vector<uint8_t*>& out_data,
                     std::vector<size_t>& out_sizes);
    
    // Stats
    struct Stats {
        size_t gpu_used;
        size_t shm_used;
        size_t gpu_hits;
        size_t shm_hits;
        size_t lustre_hits;
        size_t misses;
        size_t dedup_hits;
    };
    Stats get_stats() const;
    
    void clear();
    void flush();
    
private:
    CascadeConfig config_;
    
    std::unique_ptr<GPUBackend> gpu_;
    std::unique_ptr<ShmBackend> shm_;
    std::unique_ptr<LustreBackend> lustre_;
    
    // Dedup tracking
    ShardedIndex<bool> known_blocks_;
    ShardedIndex<bool> prefix_blocks_;
    
    // Stats
    mutable std::atomic<size_t> gpu_hits_{0};
    mutable std::atomic<size_t> shm_hits_{0};
    mutable std::atomic<size_t> lustre_hits_{0};
    mutable std::atomic<size_t> misses_{0};
    mutable std::atomic<size_t> dedup_hits_{0};
};

}  // namespace cascade
