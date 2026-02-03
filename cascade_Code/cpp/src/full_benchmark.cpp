/**
 * Full System Benchmark (C++ only)
 * 
 * Tests: GPU (vLLM-like), SHM (Cascade), Lustre (Cold), NVMe (LMCache-like)
 * Hot/Warm/Cold scenarios
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iomanip>

using namespace std::chrono;

// ============================================================================
// Configuration
// ============================================================================
struct Config {
    size_t block_size = 512 * 1024 * 1024;  // 512MB
    int num_iters = 5;
    std::string scratch_path;
    std::string shm_path = "/dev/shm";
    std::string nvme_path = "/tmp";
};

// ============================================================================
// Utilities
// ============================================================================
std::vector<uint8_t> generate_data(size_t size) {
    std::vector<uint8_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    #pragma omp parallel for
    for (size_t i = 0; i < size; i += 4096) {
        for (size_t j = i; j < std::min(i + 4096, size); j++) {
            data[j] = dis(gen);
        }
    }
    return data;
}

void drop_page_cache(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        fstat(fd, &st);
        posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
        close(fd);
    }
}

double to_gbps(size_t bytes, double seconds) {
    return bytes / seconds / (1024.0 * 1024.0 * 1024.0);
}

// ============================================================================
// 1. GPU Backend (vLLM-like)
// ============================================================================
struct GPUBench {
    void* d_buffer = nullptr;
    void* h_pinned = nullptr;
    size_t size;
    
    GPUBench(size_t sz) : size(sz) {
        cudaMalloc(&d_buffer, sz);
        cudaMallocHost(&h_pinned, sz);
    }
    
    ~GPUBench() {
        if (d_buffer) cudaFree(d_buffer);
        if (h_pinned) cudaFreeHost(h_pinned);
    }
    
    double hot_read(int iters) {
        // GPU->GPU clone
        void* d_dst;
        cudaMalloc(&d_dst, size);
        cudaDeviceSynchronize();
        
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            cudaMemcpy(d_dst, d_buffer, size, cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        cudaFree(d_dst);
        double elapsed = duration<double>(end - start).count() / iters;
        return to_gbps(size, elapsed);
    }
    
    double warm_read(const uint8_t* host_data, int iters) {
        // CPU->GPU (PCIe)
        memcpy(h_pinned, host_data, size);
        cudaDeviceSynchronize();
        
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            cudaMemcpyAsync(d_buffer, h_pinned, size, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        double elapsed = duration<double>(end - start).count() / iters;
        return to_gbps(size, elapsed);
    }
    
    double cold_read(const std::string& file_path, int iters) {
        std::vector<double> times;
        for (int i = 0; i < iters; i++) {
            drop_page_cache(file_path);
            
            auto start = high_resolution_clock::now();
            
            // Lustre -> CPU
            int fd = open(file_path.c_str(), O_RDONLY);
            read(fd, h_pinned, size);
            close(fd);
            
            // CPU -> GPU
            cudaMemcpy(d_buffer, h_pinned, size, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            
            auto end = high_resolution_clock::now();
            times.push_back(duration<double>(end - start).count());
        }
        
        double avg = 0;
        for (auto t : times) avg += t;
        return to_gbps(size, avg / times.size());
    }
};

// ============================================================================
// 2. SHM Backend (Cascade)
// ============================================================================
struct SHMBench {
    size_t size;
    std::string shm_path;
    
    SHMBench(size_t sz, const std::string& path) : size(sz), shm_path(path) {}
    
    double hot_read(const uint8_t* data, int iters) {
        // Write to SHM first
        std::string fpath = shm_path + "/cascade_hot.bin";
        int fd = open(fpath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        ftruncate(fd, size);
        void* mm = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        memcpy(mm, data, size);
        msync(mm, size, MS_SYNC);
        munmap(mm, size);
        close(fd);
        
        // Hot read from SHM
        std::vector<double> times;
        for (int i = 0; i < iters; i++) {
            fd = open(fpath.c_str(), O_RDONLY);
            void* read_mm = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
            
            auto start = high_resolution_clock::now();
            
            // Just touch memory to ensure access
            std::vector<uint8_t> out(size);
            memcpy(out.data(), read_mm, size);
            
            auto end = high_resolution_clock::now();
            times.push_back(duration<double>(end - start).count());
            
            munmap(read_mm, size);
            close(fd);
        }
        
        unlink(fpath.c_str());
        
        double avg = 0;
        for (auto t : times) avg += t;
        return to_gbps(size, avg / times.size());
    }
    
    double warm_rw(const uint8_t* data, int iters) {
        std::vector<double> times;
        
        for (int i = 0; i < iters; i++) {
            std::string fpath = shm_path + "/cascade_warm_" + std::to_string(i) + ".bin";
            
            auto start = high_resolution_clock::now();
            
            // Write
            int fd = open(fpath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            ftruncate(fd, size);
            void* mm = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            memcpy(mm, data, size);
            msync(mm, size, MS_SYNC);
            
            // Read back
            std::vector<uint8_t> out(size);
            memcpy(out.data(), mm, size);
            
            auto end = high_resolution_clock::now();
            times.push_back(duration<double>(end - start).count());
            
            munmap(mm, size);
            close(fd);
            unlink(fpath.c_str());
        }
        
        double avg = 0;
        for (auto t : times) avg += t;
        return to_gbps(size * 2, avg / times.size());  // write + read
    }
    
    double cold_read(const std::string& lustre_file, int iters) {
        std::vector<double> times;
        
        for (int i = 0; i < iters; i++) {
            drop_page_cache(lustre_file);
            
            auto start = high_resolution_clock::now();
            
            // Read from Lustre
            std::vector<uint8_t> data(size);
            int fd = open(lustre_file.c_str(), O_RDONLY);
            read(fd, data.data(), size);
            close(fd);
            
            // Write to SHM (caching)
            std::string shm_file = shm_path + "/cascade_cold_" + std::to_string(i) + ".bin";
            fd = open(shm_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            ftruncate(fd, size);
            void* mm = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            memcpy(mm, data.data(), size);
            munmap(mm, size);
            close(fd);
            
            auto end = high_resolution_clock::now();
            times.push_back(duration<double>(end - start).count());
            
            unlink(shm_file.c_str());
        }
        
        double avg = 0;
        for (auto t : times) avg += t;
        return to_gbps(size, avg / times.size());
    }
};

// ============================================================================
// 3. NVMe Backend (LMCache-like)
// ============================================================================
struct NVMeBench {
    size_t size;
    std::string nvme_path;
    
    NVMeBench(size_t sz, const std::string& path) : size(sz), nvme_path(path) {}
    
    double hot_read(const uint8_t* data, int iters) {
        // Memory copy (cached in RAM)
        std::vector<double> times;
        for (int i = 0; i < iters; i++) {
            auto start = high_resolution_clock::now();
            std::vector<uint8_t> copy(size);
            memcpy(copy.data(), data, size);
            auto end = high_resolution_clock::now();
            times.push_back(duration<double>(end - start).count());
        }
        
        double avg = 0;
        for (auto t : times) avg += t;
        return to_gbps(size, avg / times.size());
    }
    
    double warm_read(const uint8_t* data, int iters) {
        // Write to NVMe, then read back
        std::string fpath = nvme_path + "/lmcache_warm.bin";
        
        // Write first
        int fd = open(fpath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        write(fd, data, size);
        fsync(fd);
        close(fd);
        
        // Read (hot from page cache or NVMe)
        std::vector<double> times;
        for (int i = 0; i < iters; i++) {
            auto start = high_resolution_clock::now();
            std::vector<uint8_t> out(size);
            fd = open(fpath.c_str(), O_RDONLY);
            read(fd, out.data(), size);
            close(fd);
            auto end = high_resolution_clock::now();
            times.push_back(duration<double>(end - start).count());
        }
        
        unlink(fpath.c_str());
        
        double avg = 0;
        for (auto t : times) avg += t;
        return to_gbps(size, avg / times.size());
    }
    
    double cold_read(const std::string& lustre_file, int iters) {
        std::vector<double> times;
        
        for (int i = 0; i < iters; i++) {
            drop_page_cache(lustre_file);
            
            auto start = high_resolution_clock::now();
            std::vector<uint8_t> out(size);
            int fd = open(lustre_file.c_str(), O_RDONLY);
            read(fd, out.data(), size);
            close(fd);
            auto end = high_resolution_clock::now();
            
            times.push_back(duration<double>(end - start).count());
        }
        
        double avg = 0;
        for (auto t : times) avg += t;
        return to_gbps(size, avg / times.size());
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    Config cfg;
    cfg.scratch_path = getenv("SCRATCH") ? getenv("SCRATCH") : "/tmp";
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--block-size" && i+1 < argc) {
            cfg.block_size = std::stoull(argv[++i]) * 1024 * 1024;  // MB input
        }
        if (std::string(argv[i]) == "--iters" && i+1 < argc) {
            cfg.num_iters = std::stoi(argv[++i]);
        }
    }
    
    std::cout << "============================================================\n";
    std::cout << "C++ Full System Benchmark\n";
    std::cout << "Block size: " << cfg.block_size / (1024*1024) << " MB\n";
    std::cout << "Iterations: " << cfg.num_iters << "\n";
    std::cout << "============================================================\n\n";
    
    // Generate test data
    std::cout << "Generating test data...\n";
    auto data = generate_data(cfg.block_size);
    
    // Write test file to Lustre
    std::string lustre_file = cfg.scratch_path + "/bench_cold.bin";
    {
        int fd = open(lustre_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        write(fd, data.data(), data.size());
        fsync(fd);
        close(fd);
    }
    
    // Results storage
    double gpu_hot = 0, gpu_warm = 0, gpu_cold = 0;
    double shm_hot = 0, shm_warm = 0, shm_cold = 0;
    double nvme_hot = 0, nvme_warm = 0, nvme_cold = 0;
    
    // ========================================================================
    // 1. GPU Benchmark (vLLM-like)
    // ========================================================================
    std::cout << "\n=== GPU (vLLM-like) ===\n";
    {
        GPUBench gpu(cfg.block_size);
        
        // Copy data to GPU first
        cudaMemcpy(gpu.d_buffer, data.data(), cfg.block_size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        gpu_hot = gpu.hot_read(cfg.num_iters);
        std::cout << "HOT  (GPU->GPU):      " << std::fixed << std::setprecision(2) << gpu_hot << " GB/s\n";
        
        gpu_warm = gpu.warm_read(data.data(), cfg.num_iters);
        std::cout << "WARM (CPU->GPU PCIe): " << std::fixed << std::setprecision(2) << gpu_warm << " GB/s\n";
        
        gpu_cold = gpu.cold_read(lustre_file, cfg.num_iters);
        std::cout << "COLD (Lustre->GPU):   " << std::fixed << std::setprecision(2) << gpu_cold << " GB/s\n";
    }
    
    // ========================================================================
    // 2. SHM Benchmark (Cascade)
    // ========================================================================
    std::cout << "\n=== SHM (Cascade) ===\n";
    {
        SHMBench shm(cfg.block_size, cfg.shm_path);
        
        shm_hot = shm.hot_read(data.data(), cfg.num_iters);
        std::cout << "HOT  (SHM resident):  " << std::fixed << std::setprecision(2) << shm_hot << " GB/s\n";
        
        shm_warm = shm.warm_rw(data.data(), cfg.num_iters);
        std::cout << "WARM (SHM W+R):       " << std::fixed << std::setprecision(2) << shm_warm << " GB/s\n";
        
        shm_cold = shm.cold_read(lustre_file, cfg.num_iters);
        std::cout << "COLD (Lustre->SHM):   " << std::fixed << std::setprecision(2) << shm_cold << " GB/s\n";
    }
    
    // ========================================================================
    // 3. NVMe Benchmark (LMCache-like)
    // ========================================================================
    std::cout << "\n=== NVMe (LMCache-like) ===\n";
    {
        NVMeBench nvme(cfg.block_size, cfg.nvme_path);
        
        nvme_hot = nvme.hot_read(data.data(), cfg.num_iters);
        std::cout << "HOT  (Memory copy):   " << std::fixed << std::setprecision(2) << nvme_hot << " GB/s\n";
        
        nvme_warm = nvme.warm_read(data.data(), cfg.num_iters);
        std::cout << "WARM (NVMe read):     " << std::fixed << std::setprecision(2) << nvme_warm << " GB/s\n";
        
        nvme_cold = nvme.cold_read(lustre_file, cfg.num_iters);
        std::cout << "COLD (Lustre):        " << std::fixed << std::setprecision(2) << nvme_cold << " GB/s\n";
    }
    
    // Cleanup
    unlink(lustre_file.c_str());
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n============================================================\n";
    std::cout << "SUMMARY (GB/s)\n";
    std::cout << "============================================================\n";
    std::cout << std::left << std::setw(15) << "System" 
              << std::right << std::setw(12) << "HOT" 
              << std::setw(12) << "WARM" 
              << std::setw(12) << "COLD" << "\n";
    std::cout << std::string(51, '-') << "\n";
    
    std::cout << std::left << std::setw(15) << "GPU(vLLM)" 
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << gpu_hot
              << std::setw(12) << gpu_warm
              << std::setw(12) << gpu_cold << "\n";
    
    std::cout << std::left << std::setw(15) << "SHM(Cascade)" 
              << std::right << std::setw(12) << shm_hot
              << std::setw(12) << shm_warm
              << std::setw(12) << shm_cold << "\n";
    
    std::cout << std::left << std::setw(15) << "NVMe(LMCache)" 
              << std::right << std::setw(12) << nvme_hot
              << std::setw(12) << nvme_warm
              << std::setw(12) << nvme_cold << "\n";
    
    return 0;
}
