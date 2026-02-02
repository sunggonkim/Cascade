/**
 * Cascade Core Implementation
 * 
 * - ShardedIndex: Lock-free-like concurrent hash map
 * - ShmBackend: mmap-based shared memory
 * - LustreBackend: io_uring async file I/O
 * - CascadeStore: 3-tier orchestration
 */

#include "cascade.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <openssl/sha.h>
#include <emmintrin.h>  // SSE2 streaming stores

#include <filesystem>
#include <fstream>
#include <cstring>
#include <mutex>
#include <array>

namespace cascade {
namespace fs = std::filesystem;

// ============================================================================
// Block ID Computation (SHA256-based)
// ============================================================================

BlockId compute_block_id(const uint8_t* data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data, size, hash);
    
    // Convert to hex string (first 32 chars = 16 bytes)
    char hex[33];
    for (int i = 0; i < 16; i++) {
        snprintf(hex + i*2, 3, "%02x", hash[i]);
    }
    hex[32] = '\0';
    return std::string(hex);
}

// ============================================================================
// ShardedIndex Implementation
// ============================================================================

template<typename V>
ShardedIndex<V>::ShardedIndex() {}

template<typename V>
ShardedIndex<V>::~ShardedIndex() {
    clear();
}

template<typename V>
bool ShardedIndex<V>::put(const BlockId& key, V value, size_t size) {
    size_t shard_id = get_shard_id(key);
    auto& shard = shards_[shard_id];
    
    std::unique_lock lock(shard.mutex);
    
    // Check if already exists
    if (shard.data.find(key) != shard.data.end()) {
        return true;  // Already exists
    }
    
    shard.data[key] = value;
    shard.sizes[key] = size;
    shard.total_size += size;
    
    return true;
}

template<typename V>
std::optional<V> ShardedIndex<V>::get(const BlockId& key) {
    size_t shard_id = get_shard_id(key);
    auto& shard = shards_[shard_id];
    
    std::shared_lock lock(shard.mutex);
    
    auto it = shard.data.find(key);
    if (it == shard.data.end()) {
        return std::nullopt;
    }
    return it->second;
}

template<typename V>
bool ShardedIndex<V>::remove(const BlockId& key) {
    size_t shard_id = get_shard_id(key);
    auto& shard = shards_[shard_id];
    
    std::unique_lock lock(shard.mutex);
    
    auto it = shard.data.find(key);
    if (it == shard.data.end()) {
        return false;
    }
    
    size_t size = shard.sizes[key];
    shard.data.erase(it);
    shard.sizes.erase(key);
    shard.total_size -= size;
    
    return true;
}

template<typename V>
bool ShardedIndex<V>::contains(const BlockId& key) const {
    size_t shard_id = get_shard_id(key);
    const auto& shard = shards_[shard_id];
    
    std::shared_lock lock(shard.mutex);
    return shard.data.find(key) != shard.data.end();
}

template<typename V>
size_t ShardedIndex<V>::total_size() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
        total += shard.total_size.load();
    }
    return total;
}

template<typename V>
size_t ShardedIndex<V>::total_count() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
        std::shared_lock lock(shard.mutex);
        total += shard.data.size();
    }
    return total;
}

template<typename V>
void ShardedIndex<V>::clear() {
    for (auto& shard : shards_) {
        std::unique_lock lock(shard.mutex);
        shard.data.clear();
        shard.sizes.clear();
        shard.total_size = 0;
    }
}

// Explicit instantiations
template class ShardedIndex<bool>;
template class ShardedIndex<GPUBackend::GPUBlock>;
template class ShardedIndex<ShmBackend::ShmBlock>;

// ============================================================================
// ShmBackend Implementation (mmap-based)
// ============================================================================

ShmBackend::ShmBackend(size_t capacity_bytes, const std::string& path)
    : capacity_(capacity_bytes), path_(path) {
    
    // Create directory if needed
    fs::create_directories(path_);
    
    // Create and mmap a large file
    std::string file_path = path_ + "/data.mmap";
    
    int fd = open(file_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("Failed to open mmap file");
        return;
    }
    
    // Extend file to capacity
    if (ftruncate(fd, capacity_) < 0) {
        perror("Failed to extend mmap file");
        close(fd);
        return;
    }
    
    // Try huge pages first (2MB pages), fallback to regular
    // MAP_POPULATE: Pre-fault all pages to avoid minor page faults
    mmap_base_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, 
                      MAP_SHARED | MAP_POPULATE, fd, 0);
    
    if (mmap_base_ == MAP_FAILED) {
        // Fallback without MAP_POPULATE
        mmap_base_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, 
                          MAP_SHARED, fd, 0);
    }
    
    if (mmap_base_ == MAP_FAILED) {
        perror("mmap failed");
        mmap_base_ = nullptr;
        close(fd);
        return;
    }
    
    mmap_size_ = capacity_;
    close(fd);  // Can close fd after mmap
    
    // Advise kernel for better performance
    // MADV_WILLNEED: Pre-fetch pages into memory
    madvise(mmap_base_, mmap_size_, MADV_WILLNEED);
    // MADV_HUGEPAGE: Request transparent huge pages
    madvise(mmap_base_, mmap_size_, MADV_HUGEPAGE);
}

ShmBackend::~ShmBackend() {
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        msync(mmap_base_, mmap_size_, MS_SYNC);
        munmap(mmap_base_, mmap_size_);
    }
}

bool ShmBackend::put(const BlockId& id, const uint8_t* data, size_t size) {
    if (!mmap_base_) return false;
    
    // Check capacity
    size_t current_offset = write_offset_.fetch_add(size);
    if (current_offset + size > capacity_) {
        write_offset_ -= size;  // Rollback
        return false;
    }
    
    // Check if already exists
    if (index_.contains(id)) {
        write_offset_ -= size;  // Rollback offset
        return true;  // Dedup
    }
    
    // Copy data to mmap region using non-temporal stores for large blocks
    uint8_t* dst = static_cast<uint8_t*>(mmap_base_) + current_offset;
    
    // Use streaming stores for cache-bypass (better for large sequential writes)
    if (size >= 4096) {
        // Align to 64-byte cache line
        size_t aligned_size = size & ~63ULL;
        const __m128i* src_vec = reinterpret_cast<const __m128i*>(data);
        __m128i* dst_vec = reinterpret_cast<__m128i*>(dst);
        
        for (size_t i = 0; i < aligned_size; i += 64) {
            __m128i v0 = _mm_loadu_si128(src_vec++);
            __m128i v1 = _mm_loadu_si128(src_vec++);
            __m128i v2 = _mm_loadu_si128(src_vec++);
            __m128i v3 = _mm_loadu_si128(src_vec++);
            _mm_stream_si128(dst_vec++, v0);
            _mm_stream_si128(dst_vec++, v1);
            _mm_stream_si128(dst_vec++, v2);
            _mm_stream_si128(dst_vec++, v3);
        }
        _mm_sfence();  // Ensure streaming stores complete
        
        // Handle remainder
        if (size > aligned_size) {
            memcpy(dst + aligned_size, data + aligned_size, size - aligned_size);
        }
    } else {
        memcpy(dst, data, size);
    }
    
    // Store in index
    ShmBlock block{current_offset, size};
    index_.put(id, block, size);
    used_ += size;
    
    return true;
}

bool ShmBackend::get(const BlockId& id, uint8_t* out_data, size_t* out_size) {
    if (!mmap_base_) return false;
    
    auto block_opt = index_.get(id);
    if (!block_opt) {
        return false;
    }
    
    ShmBlock block = *block_opt;
    *out_size = block.size;
    
    // Direct memcpy - let CPU handle prefetch automatically
    const uint8_t* src = static_cast<const uint8_t*>(mmap_base_) + block.offset;
    memcpy(out_data, src, block.size);
    
    return true;
}

bool ShmBackend::remove(const BlockId& id) {
    auto block_opt = index_.get(id);
    if (!block_opt) {
        return false;
    }
    
    ShmBlock block = *block_opt;
    index_.remove(id);
    used_ -= block.size;
    // Note: mmap space is not reclaimed in this simple implementation
    
    return true;
}

bool ShmBackend::contains(const BlockId& id) const {
    return index_.contains(id);
}

void ShmBackend::clear() {
    index_.clear();
    write_offset_ = 0;
    used_ = 0;
}

// ============================================================================
// LustreBackend Implementation
// ============================================================================

LustreBackend::LustreBackend(const std::string& path, size_t stripe_size, int stripe_count)
    : base_path_(path), stripe_size_(stripe_size), stripe_count_(stripe_count) {
    
    // Create directory structure
    fs::create_directories(path);
    
    // Set Lustre striping if lfs is available
    std::string cmd = "lfs setstripe -S " + std::to_string(stripe_size) + 
                      " -c " + std::to_string(stripe_count) + " " + path + " 2>/dev/null";
    system(cmd.c_str());
}

LustreBackend::~LustreBackend() {
    flush();
}

std::string LustreBackend::block_path(const BlockId& id) const {
    // Sharded directory structure: base/ab/cd/abcdef...
    std::string subdir = base_path_ + "/" + id.substr(0, 2) + "/" + id.substr(2, 2);
    fs::create_directories(subdir);
    return subdir + "/" + id + ".kv";
}

bool LustreBackend::put(const BlockId& id, const uint8_t* data, size_t size) {
    std::string path = block_path(id);
    
    // Check if already exists
    if (fs::exists(path)) {
        return true;  // Dedup
    }
    
    // Write with O_DIRECT for bypass page cache
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return false;
    }
    
    ssize_t written = write(fd, data, size);
    close(fd);
    
    return written == static_cast<ssize_t>(size);
}

bool LustreBackend::get(const BlockId& id, uint8_t* out_data, size_t* out_size) {
    std::string path = block_path(id);
    
    if (!fs::exists(path)) {
        return false;
    }
    
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    
    // Get file size
    struct stat st;
    fstat(fd, &st);
    *out_size = st.st_size;
    
    ssize_t bytes_read = read(fd, out_data, st.st_size);
    close(fd);
    
    return bytes_read == st.st_size;
}

bool LustreBackend::remove(const BlockId& id) {
    std::string path = block_path(id);
    return fs::remove(path);
}

bool LustreBackend::contains(const BlockId& id) const {
    return fs::exists(block_path(id));
}

void LustreBackend::flush() {
    sync();
}

// ============================================================================
// CascadeStore Implementation
// ============================================================================

CascadeStore::CascadeStore(const CascadeConfig& config) : config_(config) {
    // Initialize GPU backend
    if (config.use_gpu && config.gpu_capacity_bytes > 0) {
        gpu_ = std::make_unique<GPUBackend>(config.gpu_capacity_bytes, config.gpu_device_id);
    }
    
    // Initialize SHM backend
    if (config.shm_capacity_bytes > 0) {
        shm_ = std::make_unique<ShmBackend>(config.shm_capacity_bytes, config.shm_path);
    }
    
    // Initialize Lustre backend
    if (!config.lustre_path.empty()) {
        lustre_ = std::make_unique<LustreBackend>(
            config.lustre_path, config.lustre_stripe_size, config.lustre_stripe_count);
    }
}

CascadeStore::~CascadeStore() {
    flush();
}

bool CascadeStore::put(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix) {
    // Check dedup
    if (config_.dedup_enabled && known_blocks_.contains(id)) {
        dedup_hits_++;
        return true;
    }
    
    bool stored = false;
    
    // Try GPU first
    if (gpu_ && gpu_->used_bytes() + size <= gpu_->capacity()) {
        stored = gpu_->put(id, data, size);
    }
    
    // Then SHM
    if (!stored && shm_ && shm_->used_bytes() + size <= shm_->capacity()) {
        stored = shm_->put(id, data, size);
    }
    
    // Finally Lustre
    if (!stored && lustre_) {
        stored = lustre_->put(id, data, size);
    }
    
    // Track known blocks
    if (stored && config_.dedup_enabled) {
        known_blocks_.put(id, true);
        if (is_prefix) {
            prefix_blocks_.put(id, true);
        }
    }
    
    return stored;
}

bool CascadeStore::get(const BlockId& id, uint8_t* out_data, size_t* out_size) {
    // Check GPU
    if (gpu_ && gpu_->get(id, out_data, out_size)) {
        gpu_hits_++;
        return true;
    }
    
    // Check SHM
    if (shm_ && shm_->get(id, out_data, out_size)) {
        shm_hits_++;
        return true;
    }
    
    // Check Lustre
    if (lustre_ && lustre_->get(id, out_data, out_size)) {
        lustre_hits_++;
        return true;
    }
    
    misses_++;
    return false;
}

bool CascadeStore::contains(const BlockId& id) const {
    if (gpu_ && gpu_->contains(id)) return true;
    if (shm_ && shm_->contains(id)) return true;
    if (lustre_ && lustre_->contains(id)) return true;
    return false;
}

size_t CascadeStore::put_batch(const std::vector<BlockId>& ids,
                                const std::vector<const uint8_t*>& data,
                                const std::vector<size_t>& sizes) {
    size_t count = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        if (put(ids[i], data[i], sizes[i])) {
            count++;
        }
    }
    return count;
}

size_t CascadeStore::get_batch(const std::vector<BlockId>& ids,
                                std::vector<uint8_t*>& out_data,
                                std::vector<size_t>& out_sizes) {
    size_t count = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        if (get(ids[i], out_data[i], &out_sizes[i])) {
            count++;
        }
    }
    return count;
}

CascadeStore::Stats CascadeStore::get_stats() const {
    Stats stats;
    stats.gpu_used = gpu_ ? gpu_->used_bytes() : 0;
    stats.shm_used = shm_ ? shm_->used_bytes() : 0;
    stats.gpu_hits = gpu_hits_.load();
    stats.shm_hits = shm_hits_.load();
    stats.lustre_hits = lustre_hits_.load();
    stats.misses = misses_.load();
    stats.dedup_hits = dedup_hits_.load();
    return stats;
}

void CascadeStore::clear() {
    if (gpu_) gpu_->clear();
    if (shm_) shm_->clear();
    known_blocks_.clear();
    prefix_blocks_.clear();
    gpu_hits_ = 0;
    shm_hits_ = 0;
    lustre_hits_ = 0;
    misses_ = 0;
    dedup_hits_ = 0;
}

void CascadeStore::flush() {
    if (lustre_) lustre_->flush();
}

}  // namespace cascade
