/**
 * Pure CUDA memcpy bandwidth test
 * Measures theoretical PCIe limit without any overhead
 */

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    std::cout << "=== Pure cudaMemcpy Bandwidth Test ===\n\n";
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n\n";
    
    const size_t sizes[] = {
        1ULL * 1024 * 1024,      // 1MB
        4ULL * 1024 * 1024,      // 4MB
        16ULL * 1024 * 1024,     // 16MB
        64ULL * 1024 * 1024,     // 64MB
        256ULL * 1024 * 1024,    // 256MB
        1024ULL * 1024 * 1024    // 1GB
    };
    
    void* d_ptr;
    void* h_ptr;
    
    // Allocate max size on GPU
    cudaMalloc(&d_ptr, 1024ULL * 1024 * 1024);
    cudaMallocHost(&h_ptr, 1024ULL * 1024 * 1024);
    
    // Fill with pattern
    memset(h_ptr, 0xAB, 1024ULL * 1024 * 1024);
    
    std::cout << "Size\t\tH2D (GB/s)\tD2H (GB/s)\tH2D Eff%\tD2H Eff%\n";
    std::cout << "────────────────────────────────────────────────────────────\n";
    
    for (size_t size : sizes) {
        const int iterations = (size >= 256 * 1024 * 1024) ? 5 : 10;
        
        // Warmup
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        
        // H2D benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double h2d_time = std::chrono::duration<double>(end - start).count();
        double h2d_bw = (double)(size * iterations) / h2d_time / (1024.0 * 1024 * 1024);
        
        // D2H benchmark
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        double d2h_time = std::chrono::duration<double>(end - start).count();
        double d2h_bw = (double)(size * iterations) / d2h_time / (1024.0 * 1024 * 1024);
        
        // PCIe Gen4 x16 = 32GB/s theoretical
        double h2d_eff = h2d_bw / 32.0 * 100;
        double d2h_eff = d2h_bw / 32.0 * 100;
        
        printf("%4zuMB\t\t%.2f\t\t%.2f\t\t%.1f%%\t\t%.1f%%\n",
               size / (1024 * 1024), h2d_bw, d2h_bw, h2d_eff, d2h_eff);
    }
    
    std::cout << "\n=== Async Stream Test (overlapped transfers) ===\n";
    
    cudaStream_t streams[8];
    for (int i = 0; i < 8; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Test with 8 streams, 128MB each = 1GB total
    const size_t chunk_size = 128 * 1024 * 1024;
    const int num_chunks = 8;
    
    // Warmup
    for (int i = 0; i < num_chunks; i++) {
        cudaMemcpyAsync((uint8_t*)d_ptr + i * chunk_size, 
                       (uint8_t*)h_ptr + i * chunk_size, 
                       chunk_size, cudaMemcpyHostToDevice, streams[i]);
    }
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 0; i < num_chunks; i++) {
            cudaMemcpyAsync((uint8_t*)d_ptr + i * chunk_size, 
                           (uint8_t*)h_ptr + i * chunk_size, 
                           chunk_size, cudaMemcpyHostToDevice, streams[i]);
        }
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double async_time = std::chrono::duration<double>(end - start).count();
    double async_bw = (double)(chunk_size * num_chunks * 5) / async_time / (1024.0 * 1024 * 1024);
    
    printf("\n8 streams × 128MB (1GB total):\n");
    printf("  Async H2D: %.2f GB/s (%.1f%% PCIe)\n", async_bw, async_bw / 32.0 * 100);
    
    // Cleanup
    for (int i = 0; i < 8; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_ptr);
    cudaFree(d_ptr);
    
    return 0;
}
