#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "chronos_matmul.h"

namespace py = pybind11;


// Forward declarations for debugging functions
std::vector<float> dequantize_block_cpp(const py::bytes& B_quantized, const std::string& qtype, ssize_t block_idx);
std::vector<float> dequantize_row_cpp(const py::bytes& B_quantized_row, const std::string& qtype, ssize_t K);

PYBIND11_MODULE(chronos_matmul, m) {
    m.doc() = "Chronos matmul with k-bit block quantization (llama.cpp style)";

    // Expose the main quantization function
    m.def("quantize", [](py::array_t<float, py::array::c_style | py::array::forcecast> B,
                         const std::string& qtype) {
        auto bufB = B.request();
        if (bufB.ndim != 2) {
            throw std::runtime_error("Input array must be 2-dimensional");
        }
        ssize_t M = bufB.shape[0];
        ssize_t K = bufB.shape[1];
        
        std::vector<char> quantized_data = quantize_model((const float*)bufB.ptr, M, K, qtype);
        
        return py::bytes(quantized_data.data(), quantized_data.size());

    }, py::arg("B"), py::arg("qtype"), "Quantizes a float32 NumPy array into a bytes object using the specified quantization type.");


    // Expose the main matrix multiplication function
    m.def("matmul_quantized", [](py::array_t<float, py::array::c_style | py::array::forcecast> A,
                                  py::bytes B_quantized,
                                  ssize_t M, // The original number of output features for B
                                  const std::string& qtype,
                                  const std::string& device) {
        auto bufA = A.request();
        if (bufA.ndim != 2) {
            throw std::runtime_error("Input array 'A' must be 2-dimensional");
        }

        ssize_t N = bufA.shape[0];
        ssize_t K = bufA.shape[1];

        char *B_ptr = PyBytes_AsString(B_quantized.ptr());
        if (B_ptr == NULL) {
            throw std::runtime_error("Failed to get pointer from bytes object");
        }

        py::array_t<float> Y({N, M});
        
        if (device == "cpu") {
            matmul_quantized_cpu((const void*)B_ptr, (const float*)bufA.ptr, (float*)Y.request().ptr, N, K, M, qtype);
        } 
#ifdef CHRONOS_USE_VULKAN
        else if (device == "vulkan") {
            matmul_quantized_vulkan((const void*)B_ptr, (const float*)bufA.ptr, (float*)Y.request().ptr, N, K, M, qtype);
        }
#endif
        else {
            std::string error_msg = "Unsupported device '" + device + "'.";
#ifndef CHRONOS_USE_VULKAN
            if (device == "vulkan") {
                error_msg += " The kernel was not compiled with Vulkan support.";
            }
#endif
            throw std::runtime_error(error_msg);
        }
        
        return Y;

    }, py::arg("A"), py::arg("B_quantized"), py::arg("M"), py::arg("qtype"), py::arg("device") = "cpu", 
       "Performs matrix multiplication Y = A @ B.T where B is a quantized bytes object. Device can be 'cpu' or 'vulkan'.");

    // Expose debugging functions
    m.def("dequantize_block", &dequantize_block_cpp,
          py::arg("B_quantized"), py::arg("qtype"), py::arg("block_idx"),
          "Dequantizes a single block from a quantized bytes object for debugging.");

    m.def("dequantize_row", &dequantize_row_cpp,
      py::arg("B_quantized_row"), py::arg("qtype"), py::arg("K"),
      "Dequantizes a single complete row from a quantized bytes object for debugging.");

#ifdef CHRONOS_USE_VULKAN
    m.attr("VULKAN_SUPPORT") = py::bool_(true);
#else
    m.attr("VULKAN_SUPPORT") = py::bool_(false);
#endif
}
