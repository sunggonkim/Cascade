/**
 * Cascade C++ Benchmark
 * 
 * Tests GPU, SHM, Lustre backends for throughput
 */

#include "cascade.hpp"

#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <omp.h>
#include <thread>

using namespace cascade;
using namespace std::chrono;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchConfig {
    size_t block_size = 128 * 1024;        // 128KB blocks
    size_t num_blocks = 1000;               // Number of blocks
    size_t num_iterations = 3;              // Iterations per test
    size_t gpu_capacity = 0;                // 0 = auto (data size + 10%)
    size_t shm_capacity = 0;                // 0 = auto (data size + 10%)
    std::string lustre_path = "/tmp/cascade_bench";
    bool test_gpu = true;
    bool test_shm = true;
    bool test_lustre = false;
    int num_threads = 8;                    // OpenMP threads
};

// ============================================================================
// Utilities
// ============================================================================

std::vector<uint8_t> generate_random_data(size_t size) {
    std::vector<uint8_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    return data;
}

template<typename T>
double benchmark_write(T& backend, const std::vector<std::vector<uint8_t>>& blocks,
                       const std::vector<BlockId>& ids, int num_threads = 8) {
    auto start = high_resolution_clock::now();
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64)
    for (size_t i = 0; i < blocks.size(); i++) {
        backend.put(ids[i], blocks[i].data(), blocks[i].size());
    }
    
    auto end = high_resolution_clock::now();
    double elapsed = duration<double>(end - start).count();
    
    size_t total_bytes = blocks.size() * blocks[0].size();
    return total_bytes / elapsed / (1024.0 * 1024 * 1024);  // GB/s
}

template<typename T>
double benchmark_read(T& backend, std::vector<std::vector<uint8_t>>& out_blocks,
                      const std::vector<BlockId>& ids, int num_threads = 8) {
    auto start = high_resolution_clock::now();
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64)
    for (size_t i = 0; i < out_blocks.size(); i++) {
        size_t size;
        backend.get(ids[i], out_blocks[i].data(), &size);
    }
    
    auto end = high_resolution_clock::now();
    double elapsed = duration<double>(end - start).count();
    
    size_t total_bytes = out_blocks.size() * out_blocks[0].size();
    return total_bytes / elapsed / (1024.0 * 1024 * 1024);
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    BenchConfig config;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--blocks" && i + 1 < argc) {
            config.num_blocks = std::stoul(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            config.block_size = std::stoul(argv[++i]) * 1024;  // KB input
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.num_iterations = std::stoul(argv[++i]);
        } else if (arg == "--gpu-only") {
            config.test_shm = false;
            config.test_lustre = false;
        } else if (arg == "--lustre") {
            config.test_lustre = true;
        } else if (arg == "--lustre-path" && i + 1 < argc) {
            config.lustre_path = argv[++i];
        }
    }
    
    omp_set_num_threads(config.num_threads);
    
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║          Cascade C++ Backend Benchmark                   ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Block size: " << std::setw(6) << config.block_size / 1024 << " KB                                   ║\n";
    std::cout << "║  Num blocks: " << std::setw(6) << config.num_blocks << "                                     ║\n";
    std::cout << "║  Total data: " << std::setw(6) << (config.block_size * config.num_blocks) / (1024*1024) << " MB                                   ║\n";
    std::cout << "║  Threads:    " << std::setw(6) << config.num_threads << "                                     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
    
    // Generate test data
    std::cout << "Generating test data..." << std::flush;
    std::vector<std::vector<uint8_t>> blocks(config.num_blocks);
    std::vector<BlockId> ids(config.num_blocks);
    
    for (size_t i = 0; i < config.num_blocks; i++) {
        blocks[i] = generate_random_data(config.block_size);
        ids[i] = compute_block_id(blocks[i].data(), blocks[i].size());
    }
    std::cout << " done\n\n";
    
    // Auto-size capacities based on data size (add 20% margin)
    size_t total_data_size = config.block_size * config.num_blocks;
    if (config.gpu_capacity == 0) {
        config.gpu_capacity = (size_t)(total_data_size * 1.2);
    }
    if (config.shm_capacity == 0) {
        config.shm_capacity = (size_t)(total_data_size * 1.2);
    }
    
    std::cout << "  Actual GPU capacity: " << config.gpu_capacity / (1024*1024) << " MB\n";
    std::cout << "  Actual SHM capacity: " << config.shm_capacity / (1024*1024) << " MB\n\n";
    
    // Output buffers
    std::vector<std::vector<uint8_t>> out_blocks(config.num_blocks,
        std::vector<uint8_t>(config.block_size));
    
    // ========================================================================
    // GPU Backend Benchmark
    // ========================================================================
    if (config.test_gpu) {
        std::cout << "═══════════════════════════════════════════════════════════\n";
        std::cout << "  GPU Backend (CUDA + Pinned Memory + Streams)\n";
        std::cout << "═══════════════════════════════════════════════════════════\n";
        
        GPUBackend gpu(config.gpu_capacity, 0);
        
        double write_total = 0, read_total = 0;
        for (size_t iter = 0; iter < config.num_iterations; iter++) {
            gpu.clear();
            
            double write_rate = benchmark_write(gpu, blocks, ids);
            double read_rate = benchmark_read(gpu, out_blocks, ids);
            
            std::cout << "  Iter " << iter + 1 << ": Write " << std::fixed << std::setprecision(2)
                      << write_rate << " GB/s, Read " << read_rate << " GB/s\n";
            
            write_total += write_rate;
            read_total += read_rate;
        }
        
        std::cout << "  ─────────────────────────────────────────────────────────\n";
        std::cout << "  Average: Write " << std::fixed << std::setprecision(2)
                  << write_total / config.num_iterations << " GB/s, Read "
                  << read_total / config.num_iterations << " GB/s\n";
        std::cout << "  PCIe Efficiency: Write " 
                  << (write_total / config.num_iterations / 32.0 * 100) << "%, Read "
                  << (read_total / config.num_iterations / 32.0 * 100) << "%\n\n";
    }
    
    // ========================================================================
    // SHM Backend Benchmark
    // ========================================================================
    if (config.test_shm) {
        std::cout << "═══════════════════════════════════════════════════════════\n";
        std::cout << "  SHM Backend (mmap + lock-free index)\n";
        std::cout << "═══════════════════════════════════════════════════════════\n";
        
        ShmBackend shm(config.shm_capacity, "/dev/shm/cascade_bench");
        
        double write_total = 0, read_total = 0;
        for (size_t iter = 0; iter < config.num_iterations; iter++) {
            shm.clear();
            
            double write_rate = benchmark_write(shm, blocks, ids);
            double read_rate = benchmark_read(shm, out_blocks, ids);
            
            std::cout << "  Iter " << iter + 1 << ": Write " << std::fixed << std::setprecision(2)
                      << write_rate << " GB/s, Read " << read_rate << " GB/s\n";
            
            write_total += write_rate;
            read_total += read_rate;
        }
        
        std::cout << "  ─────────────────────────────────────────────────────────\n";
        std::cout << "  Average: Write " << std::fixed << std::setprecision(2)
                  << write_total / config.num_iterations << " GB/s, Read "
                  << read_total / config.num_iterations << " GB/s\n";
        std::cout << "  DDR4 Efficiency: Write "
                  << (write_total / config.num_iterations / 204.0 * 100) << "%, Read "
                  << (read_total / config.num_iterations / 204.0 * 100) << "%\n\n";
    }
    
    // ========================================================================
    // Lustre Backend Benchmark
    // ========================================================================
    if (config.test_lustre) {
        std::cout << "═══════════════════════════════════════════════════════════\n";
        std::cout << "  Lustre Backend (parallel file I/O)\n";
        std::cout << "═══════════════════════════════════════════════════════════\n";
        
        LustreBackend lustre(config.lustre_path, 4 * 1024 * 1024, 4);
        
        double write_total = 0, read_total = 0;
        for (size_t iter = 0; iter < config.num_iterations; iter++) {
            // Clear by removing directory (expensive)
            system(("rm -rf " + config.lustre_path + "/*").c_str());
            
            double write_rate = benchmark_write(lustre, blocks, ids);
            lustre.flush();
            double read_rate = benchmark_read(lustre, out_blocks, ids);
            
            std::cout << "  Iter " << iter + 1 << ": Write " << std::fixed << std::setprecision(2)
                      << write_rate << " GB/s, Read " << read_rate << " GB/s\n";
            
            write_total += write_rate;
            read_total += read_rate;
        }
        
        std::cout << "  ─────────────────────────────────────────────────────────\n";
        std::cout << "  Average: Write " << std::fixed << std::setprecision(2)
                  << write_total / config.num_iterations << " GB/s, Read "
                  << read_total / config.num_iterations << " GB/s\n\n";
    }
    
    std::cout << "Benchmark complete.\n";
    return 0;
}
