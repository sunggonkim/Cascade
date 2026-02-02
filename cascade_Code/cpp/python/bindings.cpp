/**
 * Cascade Python Bindings (pybind11)
 * 
 * Exposes C++ CascadeStore to Python with numpy/CuPy interop
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cascade.hpp"

namespace py = pybind11;

// Helper: Convert numpy array to raw pointer
template<typename T>
const T* numpy_data(py::array_t<T>& arr) {
    return static_cast<const T*>(arr.request().ptr);
}

template<typename T>
T* numpy_mutable_data(py::array_t<T>& arr) {
    return static_cast<T*>(arr.request().ptr);
}

PYBIND11_MODULE(cascade_cpp, m) {
    m.doc() = "High-performance Cascade KV Cache for LLM inference";
    
    // ========================================================================
    // CascadeConfig
    // ========================================================================
    py::class_<cascade::CascadeConfig>(m, "CascadeConfig")
        .def(py::init<>())
        .def_readwrite("gpu_capacity_bytes", &cascade::CascadeConfig::gpu_capacity_bytes)
        .def_readwrite("shm_capacity_bytes", &cascade::CascadeConfig::shm_capacity_bytes)
        .def_readwrite("shm_path", &cascade::CascadeConfig::shm_path)
        .def_readwrite("lustre_path", &cascade::CascadeConfig::lustre_path)
        .def_readwrite("lustre_stripe_size", &cascade::CascadeConfig::lustre_stripe_size)
        .def_readwrite("lustre_stripe_count", &cascade::CascadeConfig::lustre_stripe_count)
        .def_readwrite("gpu_device_id", &cascade::CascadeConfig::gpu_device_id)
        .def_readwrite("dedup_enabled", &cascade::CascadeConfig::dedup_enabled)
        .def_readwrite("compression_enabled", &cascade::CascadeConfig::compression_enabled)
        .def_readwrite("use_gpu", &cascade::CascadeConfig::use_gpu);
    
    // ========================================================================
    // CascadeStore::Stats
    // ========================================================================
    py::class_<cascade::CascadeStore::Stats>(m, "CascadeStats")
        .def_readonly("gpu_used", &cascade::CascadeStore::Stats::gpu_used)
        .def_readonly("shm_used", &cascade::CascadeStore::Stats::shm_used)
        .def_readonly("gpu_hits", &cascade::CascadeStore::Stats::gpu_hits)
        .def_readonly("shm_hits", &cascade::CascadeStore::Stats::shm_hits)
        .def_readonly("lustre_hits", &cascade::CascadeStore::Stats::lustre_hits)
        .def_readonly("misses", &cascade::CascadeStore::Stats::misses)
        .def_readonly("dedup_hits", &cascade::CascadeStore::Stats::dedup_hits)
        .def("__repr__", [](const cascade::CascadeStore::Stats& s) {
            return "CascadeStats(gpu=" + std::to_string(s.gpu_used / (1024*1024)) + "MB"
                   ", shm=" + std::to_string(s.shm_used / (1024*1024)) + "MB"
                   ", hits=" + std::to_string(s.gpu_hits + s.shm_hits + s.lustre_hits) +
                   ", dedup=" + std::to_string(s.dedup_hits) + ")";
        });
    
    // ========================================================================
    // CascadeStore (Main Interface)
    // ========================================================================
    py::class_<cascade::CascadeStore>(m, "CascadeStore")
        .def(py::init<const cascade::CascadeConfig&>())
        
        // put: numpy array → store
        .def("put", [](cascade::CascadeStore& self, 
                       const std::string& block_id,
                       py::array_t<uint8_t>& data,
                       bool is_prefix) {
            py::buffer_info buf = data.request();
            return self.put(block_id, 
                           static_cast<const uint8_t*>(buf.ptr),
                           buf.size,
                           is_prefix);
        }, py::arg("block_id"), py::arg("data"), py::arg("is_prefix") = false,
           "Store a KV cache block")
        
        // get: store → numpy array
        .def("get", [](cascade::CascadeStore& self,
                       const std::string& block_id,
                       py::array_t<uint8_t>& out_data) {
            py::buffer_info buf = out_data.request();
            size_t size = 0;
            bool found = self.get(block_id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        }, py::arg("block_id"), py::arg("out_data"),
           "Retrieve a KV cache block")
        
        // put_batch: batch storage for throughput
        .def("put_batch", [](cascade::CascadeStore& self,
                             const std::vector<std::string>& ids,
                             py::list data_list) {
            std::vector<const uint8_t*> data_ptrs;
            std::vector<size_t> sizes;
            
            for (auto item : data_list) {
                auto arr = item.cast<py::array_t<uint8_t>>();
                py::buffer_info buf = arr.request();
                data_ptrs.push_back(static_cast<const uint8_t*>(buf.ptr));
                sizes.push_back(buf.size);
            }
            
            return self.put_batch(ids, data_ptrs, sizes);
        }, py::arg("ids"), py::arg("data_list"),
           "Store multiple KV cache blocks in batch")
        
        // get_batch: batch retrieval
        .def("get_batch", [](cascade::CascadeStore& self,
                             const std::vector<std::string>& ids,
                             py::list out_list) {
            std::vector<uint8_t*> out_ptrs;
            std::vector<size_t> sizes(ids.size(), 0);
            
            for (auto item : out_list) {
                auto arr = item.cast<py::array_t<uint8_t>>();
                py::buffer_info buf = arr.request();
                out_ptrs.push_back(static_cast<uint8_t*>(buf.ptr));
            }
            
            size_t count = self.get_batch(ids, out_ptrs, sizes);
            return py::make_tuple(count, sizes);
        }, py::arg("ids"), py::arg("out_list"),
           "Retrieve multiple KV cache blocks in batch")
        
        // contains: check existence
        .def("contains", &cascade::CascadeStore::contains,
             py::arg("block_id"), "Check if block exists")
        
        // stats
        .def("get_stats", &cascade::CascadeStore::get_stats, "Get store statistics")
        
        // clear
        .def("clear", &cascade::CascadeStore::clear, "Clear all backends")
        
        // flush
        .def("flush", &cascade::CascadeStore::flush, "Flush to persistent storage");
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    m.def("compute_block_id", [](py::array_t<uint8_t>& data) {
        py::buffer_info buf = data.request();
        return cascade::compute_block_id(static_cast<const uint8_t*>(buf.ptr), buf.size);
    }, py::arg("data"), "Compute SHA256-based block ID from data");
    
    // ========================================================================
    // GPU Backend Direct Access (for benchmarks)
    // ========================================================================
    py::class_<cascade::GPUBackend>(m, "GPUBackend")
        .def(py::init<size_t, int>(), 
             py::arg("capacity_bytes"), py::arg("device_id") = 0)
        .def("put", [](cascade::GPUBackend& self,
                       const std::string& id,
                       py::array_t<uint8_t>& data) {
            py::buffer_info buf = data.request();
            return self.put(id, static_cast<const uint8_t*>(buf.ptr), buf.size);
        })
        .def("get", [](cascade::GPUBackend& self,
                       const std::string& id,
                       py::array_t<uint8_t>& out) {
            py::buffer_info buf = out.request();
            size_t size = 0;
            bool found = self.get(id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::GPUBackend::contains)
        .def("capacity", &cascade::GPUBackend::capacity)
        .def("used_bytes", &cascade::GPUBackend::used_bytes)
        .def("clear", &cascade::GPUBackend::clear);
    
    // ========================================================================
    // SHM Backend Direct Access
    // ========================================================================
    py::class_<cascade::ShmBackend>(m, "ShmBackend")
        .def(py::init<size_t, const std::string&>(),
             py::arg("capacity_bytes"), py::arg("path") = "/dev/shm/cascade")
        .def("put", [](cascade::ShmBackend& self,
                       const std::string& id,
                       py::array_t<uint8_t>& data) {
            py::buffer_info buf = data.request();
            return self.put(id, static_cast<const uint8_t*>(buf.ptr), buf.size);
        })
        .def("get", [](cascade::ShmBackend& self,
                       const std::string& id,
                       py::array_t<uint8_t>& out) {
            py::buffer_info buf = out.request();
            size_t size = 0;
            bool found = self.get(id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::ShmBackend::contains)
        .def("capacity", &cascade::ShmBackend::capacity)
        .def("used_bytes", &cascade::ShmBackend::used_bytes)
        .def("clear", &cascade::ShmBackend::clear);
    
    // ========================================================================
    // Lustre Backend Direct Access
    // ========================================================================
    py::class_<cascade::LustreBackend>(m, "LustreBackend")
        .def(py::init<const std::string&, size_t, int>(),
             py::arg("path"), 
             py::arg("stripe_size") = 4 * 1024 * 1024,
             py::arg("stripe_count") = 4)
        .def("put", [](cascade::LustreBackend& self,
                       const std::string& id,
                       py::array_t<uint8_t>& data) {
            py::buffer_info buf = data.request();
            return self.put(id, static_cast<const uint8_t*>(buf.ptr), buf.size);
        })
        .def("get", [](cascade::LustreBackend& self,
                       const std::string& id,
                       py::array_t<uint8_t>& out) {
            py::buffer_info buf = out.request();
            size_t size = 0;
            bool found = self.get(id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::LustreBackend::contains)
        .def("flush", &cascade::LustreBackend::flush);
}
