/**
 * Cascade GPU Backend - CUDA Implementation
 * 
 * Optimizations:
 * 1. GPU Memory Pool - Pre-allocated, offset-based
 * 2. Multiple Pinned Buffers - Thread-local staging
 * 3. 8 CUDA Streams - Maximum concurrency
 * 4. Async batch sync - Pipeline parallelism
 * 
 * Target: 25+ GB/s on PCIe Gen4
 */

#include "cascade.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cascade {

// ============================================================================
// Constants
// ============================================================================

static constexpr int NUM_STREAMS = 8;
static constexpr size_t PINNED_BUFFER_SIZE = 64 * 1024 * 1024;   // 64MB per buffer
static constexpr int NUM_PINNED_BUFFERS = 32;  // 32 thread-local buffers = 2GB total

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                               \
            return false;                                                   \
        }                                                                   \
    } while (0)

// ============================================================================
// GPU Memory Pool - Avoids cudaMalloc overhead
// ============================================================================

class GPUMemoryPool {
public:
    GPUMemoryPool(size_t total_size, int device_id) : capacity_(total_size) {
        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc(&pool_base_, total_size);
        if (err != cudaSuccess) {
            pool_base_ = nullptr;
            fprintf(stderr, "Failed to allocate GPU memory pool\n");
        }
    }
    
    ~GPUMemoryPool() {
        if (pool_base_) {
            cudaFree(pool_base_);
        }
    }
    
    // Simple bump allocator (fast, no fragmentation handling)
    void* alloc(size_t size) {
        size_t aligned_size = (size + 255) & ~255;  // 256-byte alignment
        size_t offset = current_offset_.fetch_add(aligned_size);
        if (offset + aligned_size > capacity_) {
            current_offset_ -= aligned_size;  // Rollback
            return nullptr;
        }
        return static_cast<uint8_t*>(pool_base_) + offset;
    }
    
    void reset() {
        current_offset_ = 0;
    }
    
    size_t used() const { return current_offset_.load(); }
    
private:
    void* pool_base_ = nullptr;
    size_t capacity_;
    std::atomic<size_t> current_offset_{0};
};

// ============================================================================
// GPUBackend Implementation with Memory Pool
// ============================================================================

GPUBackend::GPUBackend(size_t capacity_bytes, int device_id)
    : capacity_(capacity_bytes), device_id_(device_id) {
    init_cuda();
}

GPUBackend::~GPUBackend() {
    // Free pinned buffers
    for (int i = 0; i < NUM_PINNED_BUFFERS; i++) {
        if (pinned_buffers_[i]) {
            cudaFreeHost(pinned_buffers_[i]);
        }
    }
    // Free streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (cuda_streams_[i]) {
            cudaStreamDestroy(static_cast<cudaStream_t>(cuda_streams_[i]));
        }
    }
    clear();
}

bool GPUBackend::init_cuda() {
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device %d\n", device_id_);
        return false;
    }
    
    // Allocate multiple pinned staging buffers (256MB each)
    for (int i = 0; i < NUM_PINNED_BUFFERS; i++) {
        err = cudaHostAlloc(&pinned_buffers_[i], PINNED_BUFFER_SIZE, 
                            cudaHostAllocDefault | cudaHostAllocPortable);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate pinned buffer %d\n", i);
            return false;
        }
    }
    pinned_buffer_ = pinned_buffers_[0];  // Default
    pinned_size_ = PINNED_BUFFER_SIZE;
    
    // Create 8 streams for maximum concurrency
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStream_t stream;
        err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to create CUDA stream %d\n", i);
            return false;
        }
        cuda_streams_[i] = stream;
    }
    
    // Initialize GPU memory pool
    memory_pool_ = std::make_unique<GPUMemoryPool>(capacity_, device_id_);
    
    return true;
}

void* GPUBackend::alloc_gpu(size_t size) {
    if (memory_pool_) {
        return memory_pool_->alloc(size);
    }
    // Fallback to regular allocation
    void* ptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void GPUBackend::free_gpu(void* ptr) {
    // With memory pool, we don't free individual blocks
    // Memory is reclaimed on clear()
}

void GPUBackend::copy_h2d_async(void* dst, const void* src, size_t size, int stream) {
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 
                    static_cast<cudaStream_t>(cuda_streams_[stream % NUM_STREAMS]));
}

void GPUBackend::copy_d2h_async(void* dst, const void* src, size_t size, int stream) {
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(cuda_streams_[stream % NUM_STREAMS]));
}

void GPUBackend::sync_stream(int stream) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_streams_[stream % NUM_STREAMS]));
}

bool GPUBackend::put(const BlockId& id, const uint8_t* data, size_t size) {
    // Check capacity
    if (used_.load() + size > capacity_) {
        return false;
    }
    
    // Check if already exists (dedup)
    if (index_.contains(id)) {
        return true;
    }
    
    // Allocate from memory pool (very fast)
    void* gpu_ptr = alloc_gpu(size);
    if (!gpu_ptr) {
        return false;
    }
    
    // Get thread-local stream (using OpenMP tid for stability)
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = current_stream_.fetch_add(1);
    #endif
    int stream_id = tid % NUM_STREAMS;
    int buf_id = tid % NUM_PINNED_BUFFERS;
    void* pinned = pinned_buffers_[buf_id];
    
    // Copy via pinned buffer for small blocks
    if (size <= PINNED_BUFFER_SIZE) {
        memcpy(pinned, data, size);
        copy_h2d_async(gpu_ptr, pinned, size, stream_id);
        sync_stream(stream_id);
    } else {
        // Large block: direct copy
        cudaMemcpy(gpu_ptr, data, size, cudaMemcpyHostToDevice);
    }
    
    // Store in index
    GPUBlock block{gpu_ptr, size};
    index_.put(id, block, size);
    used_ += size;
    
    return true;
}

bool GPUBackend::get(const BlockId& id, uint8_t* out_data, size_t* out_size) {
    auto block_opt = index_.get(id);
    if (!block_opt) {
        return false;
    }
    
    GPUBlock block = *block_opt;
    *out_size = block.size;
    
    // Get thread-local stream
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = current_stream_.fetch_add(1);
    #endif
    int stream_id = tid % NUM_STREAMS;
    int buf_id = tid % NUM_PINNED_BUFFERS;
    void* pinned = pinned_buffers_[buf_id];
    
    // Copy back via pinned buffer
    if (block.size <= PINNED_BUFFER_SIZE) {
        copy_d2h_async(pinned, block.ptr, block.size, stream_id);
        sync_stream(stream_id);
        memcpy(out_data, pinned, block.size);
    } else {
        cudaMemcpy(out_data, block.ptr, block.size, cudaMemcpyDeviceToHost);
    }
    
    return true;
}

bool GPUBackend::remove(const BlockId& id) {
    auto block_opt = index_.get(id);
    if (!block_opt) {
        return false;
    }
    
    GPUBlock block = *block_opt;
    // With pool, we don't actually free - space reclaimed on clear()
    index_.remove(id);
    used_ -= block.size;
    
    return true;
}

bool GPUBackend::contains(const BlockId& id) const {
    return index_.contains(id);
}

void GPUBackend::clear() {
    index_.clear();
    used_ = 0;
    if (memory_pool_) {
        memory_pool_->reset();
    }
}

}  // namespace cascade
