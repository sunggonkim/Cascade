/**
 * High-Performance KV Cache Aggregator
 * 
 * C++ + MPI + OpenMP for ultra-fast KV cache processing on HPC.
 * Reads raw KV blocks and writes aggregated Lustre-optimized files.
 * 
 * Build:
 *   module load PrgEnv-gnu cray-mpich
 *   CC -O3 -fopenmp -o kv_aggregator kv_aggregator.cpp -lcrypto
 * 
 * Run:
 *   srun -N 32 -n 128 ./kv_aggregator /input/raw /output/aggregated
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <atomic>

#ifndef NO_MPI
#include <mpi.h>
#endif
#include <omp.h>
#include <openssl/sha.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

// LLaMA-70B KV cache constants
constexpr int NUM_LAYERS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_SIZE = 256;  // tokens per block
constexpr size_t BYTES_PER_TOKEN = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * sizeof(uint16_t);
constexpr size_t BYTES_PER_BLOCK = BLOCK_SIZE * BYTES_PER_TOKEN;

// Aggregation config
constexpr int BLOCKS_PER_FILE = 256;
constexpr size_t FILE_SIZE = BLOCKS_PER_FILE * (16 + BYTES_PER_BLOCK * 2);  // header + key + value
constexpr char MAGIC[] = "CASKV001";

struct BlockIndex {
    char block_id[32];
    uint64_t offset;
    uint64_t size;
};

class ContentHasher {
public:
    static std::string compute(const void* key_data, size_t key_size,
                               const void* value_data, size_t value_size) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        SHA256_Update(&ctx, key_data, key_size);
        SHA256_Update(&ctx, value_data, value_size);
        SHA256_Final(hash, &ctx);
        
        char hex[33];
        for (int i = 0; i < 16; i++) {
            sprintf(hex + i * 2, "%02x", hash[i]);
        }
        hex[32] = '\0';
        return std::string(hex);
    }
};

class AggregatedWriter {
private:
    fs::path output_dir_;
    int rank_;
    int file_id_ = 0;
    int blocks_in_file_ = 0;
    std::ofstream current_file_;
    std::vector<BlockIndex> current_index_;
    std::atomic<uint64_t> total_blocks_{0};
    std::mutex write_mutex_;
    
    // Direct I/O buffer (aligned for O_DIRECT)
    static constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024;  // 4MB aligned
    alignas(4096) char* buffer_ = nullptr;
    size_t buffer_pos_ = 0;
    
public:
    AggregatedWriter(const fs::path& output_dir, int rank)
        : output_dir_(output_dir), rank_(rank) {
        
        // Create rank directory
        fs::path rank_dir = output_dir_ / ("rank_" + std::to_string(rank_));
        fs::create_directories(rank_dir);
        
        // Allocate aligned buffer for Lustre
        buffer_ = static_cast<char*>(aligned_alloc(4096, BUFFER_SIZE));
        
        // Set Lustre stripe
        std::string cmd = "lfs setstripe -c 16 -S 4m " + rank_dir.string() + " 2>/dev/null";
        system(cmd.c_str());
    }
    
    ~AggregatedWriter() {
        close_current_file();
        free(buffer_);
    }
    
    void open_new_file() {
        close_current_file();
        
        fs::path rank_dir = output_dir_ / ("rank_" + std::to_string(rank_));
        std::ostringstream ss;
        ss << "agg_" << std::setw(6) << std::setfill('0') << file_id_ << ".bin";
        fs::path filepath = rank_dir / ss.str();
        
        current_file_.open(filepath, std::ios::binary | std::ios::trunc);
        
        // Write magic header
        current_file_.write(MAGIC, 8);
        uint32_t block_count = 0;
        current_file_.write(reinterpret_cast<char*>(&block_count), 4);
        
        current_index_.clear();
        blocks_in_file_ = 0;
        file_id_++;
        buffer_pos_ = 0;
    }
    
    void close_current_file() {
        if (!current_file_.is_open()) return;
        
        // Flush buffer
        flush_buffer();
        
        // Update block count in header
        current_file_.seekp(8);
        uint32_t block_count = static_cast<uint32_t>(blocks_in_file_);
        current_file_.write(reinterpret_cast<char*>(&block_count), 4);
        
        // Write index at end
        current_file_.seekp(0, std::ios::end);
        uint64_t index_offset = current_file_.tellp();
        
        for (const auto& idx : current_index_) {
            current_file_.write(idx.block_id, 32);
            current_file_.write(reinterpret_cast<const char*>(&idx.offset), 8);
            current_file_.write(reinterpret_cast<const char*>(&idx.size), 8);
        }
        
        // Write index offset
        current_file_.write(reinterpret_cast<char*>(&index_offset), 8);
        
        current_file_.close();
    }
    
    void flush_buffer() {
        if (buffer_pos_ > 0 && current_file_.is_open()) {
            current_file_.write(buffer_, buffer_pos_);
            buffer_pos_ = 0;
        }
    }
    
    void write_block(const std::string& block_id,
                     const void* key_data, size_t key_size,
                     const void* value_data, size_t value_size) {
        std::lock_guard<std::mutex> lock(write_mutex_);
        
        if (!current_file_.is_open() || blocks_in_file_ >= BLOCKS_PER_FILE) {
            open_new_file();
        }
        
        uint64_t offset = current_file_.tellp();
        
        // Write block header
        uint64_t sizes[2] = {key_size, value_size};
        current_file_.write(reinterpret_cast<char*>(sizes), 16);
        
        // Write data
        current_file_.write(static_cast<const char*>(key_data), key_size);
        current_file_.write(static_cast<const char*>(value_data), value_size);
        
        // Update index
        BlockIndex idx;
        std::memcpy(idx.block_id, block_id.c_str(), 32);
        idx.offset = offset;
        idx.size = 16 + key_size + value_size;
        current_index_.push_back(idx);
        
        blocks_in_file_++;
        total_blocks_++;
    }
    
    uint64_t get_total_blocks() const { return total_blocks_.load(); }
};

class KVGenerator {
    /**
     * Generates KV cache blocks with realistic patterns.
     * This simulates prefix sharing (common system prompts)
     * that would occur in production LLM serving.
     */
private:
    int rank_;
    int world_size_;
    std::vector<std::vector<uint16_t>> prefix_templates_;
    std::mt19937 rng_;
    
public:
    KVGenerator(int rank, int world_size)
        : rank_(rank), world_size_(world_size), rng_(rank * 42) {
        
        // Generate common prefix templates (simulating shared system prompts)
        int num_prefixes = 100;
        for (int i = 0; i < num_prefixes; i++) {
            std::vector<uint16_t> prefix(BYTES_PER_BLOCK / sizeof(uint16_t));
            std::uniform_int_distribution<uint16_t> dist(0, 65535);
            for (auto& v : prefix) {
                v = dist(rng_);
            }
            prefix_templates_.push_back(prefix);
        }
    }
    
    void generate_block(int session_id, int block_idx,
                        std::vector<uint16_t>& key_out,
                        std::vector<uint16_t>& value_out) {
        
        size_t block_elements = BYTES_PER_BLOCK / sizeof(uint16_t);
        key_out.resize(block_elements);
        value_out.resize(block_elements);
        
        // First block uses prefix template (simulating shared system prompt)
        if (block_idx == 0) {
            int prefix_idx = session_id % prefix_templates_.size();
            std::copy(prefix_templates_[prefix_idx].begin(),
                      prefix_templates_[prefix_idx].end(),
                      key_out.begin());
            std::copy(prefix_templates_[prefix_idx].begin(),
                      prefix_templates_[prefix_idx].end(),
                      value_out.begin());
        } else {
            // Subsequent blocks are session-specific
            std::uniform_int_distribution<uint16_t> dist(0, 65535);
            for (size_t i = 0; i < block_elements; i++) {
                key_out[i] = dist(rng_);
                value_out[i] = dist(rng_);
            }
        }
    }
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --output DIR      Output directory (required)\n"
              << "  --sessions N      Number of sessions per rank (default: 1000)\n"
              << "  --blocks N        Blocks per session (default: 4)\n"
              << "  --help            Show this help\n";
}

int main(int argc, char** argv) {
#ifndef NO_MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
    int rank = 0, world_size = 1;
#endif
    
    // Parse arguments
    std::string output_dir;
    int sessions_per_rank = 1000;
    int blocks_per_session = 4;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--sessions" && i + 1 < argc) {
            sessions_per_rank = std::stoi(argv[++i]);
        } else if (arg == "--blocks" && i + 1 < argc) {
            blocks_per_session = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            if (rank == 0) print_usage(argv[0]);
#ifndef NO_MPI
            MPI_Finalize();
#endif
            return 0;
        }
    }
    
    if (output_dir.empty()) {
        if (rank == 0) {
            std::cerr << "Error: --output is required\n";
            print_usage(argv[0]);
        }
#ifndef NO_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "============================================\n";
        std::cout << "KV Cache Aggregator - HPC Optimized\n";
        std::cout << "============================================\n";
        std::cout << "Output: " << output_dir << "\n";
        std::cout << "World size: " << world_size << "\n";
        std::cout << "Sessions per rank: " << sessions_per_rank << "\n";
        std::cout << "Blocks per session: " << blocks_per_session << "\n";
        std::cout << "Block size: " << BYTES_PER_BLOCK / (1024*1024) << " MB\n";
        std::cout << "Expected total: " 
                  << (double)(world_size * sessions_per_rank * blocks_per_session * BYTES_PER_BLOCK * 2) / (1024.0*1024*1024)
                  << " GB\n";
        std::cout << "============================================\n";
        
        fs::create_directories(output_dir);
    }
    
#ifndef NO_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create writer and generator
    AggregatedWriter writer(output_dir, rank);
    KVGenerator generator(rank, world_size);
    
    std::vector<uint16_t> key_data, value_data;
    std::unordered_set<std::string> seen_hashes;
    uint64_t unique_blocks = 0;
    uint64_t duplicate_blocks = 0;
    
    // Generate and write blocks
    #pragma omp parallel for schedule(dynamic) \
        private(key_data, value_data) \
        reduction(+:unique_blocks, duplicate_blocks)
    for (int session = 0; session < sessions_per_rank; session++) {
        for (int block = 0; block < blocks_per_session; block++) {
            generator.generate_block(session, block, key_data, value_data);
            
            // Compute content hash
            std::string hash = ContentHasher::compute(
                key_data.data(), key_data.size() * sizeof(uint16_t),
                value_data.data(), value_data.size() * sizeof(uint16_t)
            );
            
            // Check for dedup
            bool is_new = false;
            #pragma omp critical
            {
                if (seen_hashes.find(hash) == seen_hashes.end()) {
                    seen_hashes.insert(hash);
                    is_new = true;
                }
            }
            
            if (is_new) {
                writer.write_block(
                    hash,
                    key_data.data(), key_data.size() * sizeof(uint16_t),
                    value_data.data(), value_data.size() * sizeof(uint16_t)
                );
                unique_blocks++;
            } else {
                duplicate_blocks++;
            }
        }
        
        if (rank == 0 && session % 100 == 0) {
            std::cout << "\rProgress: " << session << "/" << sessions_per_rank << std::flush;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    // Gather statistics
    uint64_t total_unique = 0, total_duplicate = 0;
    double max_elapsed = 0;
#ifndef NO_MPI
    MPI_Reduce(&unique_blocks, &total_unique, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&duplicate_blocks, &total_duplicate, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    total_unique = unique_blocks;
    total_duplicate = duplicate_blocks;
    max_elapsed = elapsed;
#endif
    
    if (rank == 0) {
        double total_bytes = (double)total_unique * BYTES_PER_BLOCK * 2;
        double throughput = total_bytes / max_elapsed / (1024.0 * 1024 * 1024);
        
        std::cout << "\n============================================\n";
        std::cout << "Complete!\n";
        std::cout << "============================================\n";
        std::cout << "Unique blocks: " << total_unique << "\n";
        std::cout << "Duplicate blocks: " << total_duplicate << "\n";
        std::cout << "Dedup ratio: " << (double)total_duplicate / (total_unique + total_duplicate) * 100 << "%\n";
        std::cout << "Total data: " << total_bytes / (1024.0*1024*1024) << " GB\n";
        std::cout << "Time: " << max_elapsed << " s\n";
        std::cout << "Throughput: " << throughput << " GB/s\n";
        std::cout << "============================================\n";
        
        // Write global summary
        fs::path summary_path = fs::path(output_dir) / "summary.json";
        std::ofstream summary(summary_path);
        summary << "{\n";
        summary << "  \"unique_blocks\": " << total_unique << ",\n";
        summary << "  \"duplicate_blocks\": " << total_duplicate << ",\n";
        summary << "  \"total_bytes\": " << (uint64_t)total_bytes << ",\n";
        summary << "  \"elapsed_seconds\": " << max_elapsed << ",\n";
        summary << "  \"throughput_gbps\": " << throughput << ",\n";
        summary << "  \"world_size\": " << world_size << "\n";
        summary << "}\n";
    }
    
#ifndef NO_MPI
    MPI_Finalize();
#endif
    return 0;
}
