#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>
#include <optional>
#include <pybind11/pybind11.h> // For py::bytes

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#include <intrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

// --- Architecture detection for SIMD instructions ---
#if (defined(__AVX512F__) && defined(__AVX512BW__))
    #include <immintrin.h>
    #define HIERARCHOS_USE_AVX512 1
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define HIERARCHOS_USE_AVX2 1
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define HIERARCHOS_USE_NEON 1
#endif


// --- Common K-Quants Block Size ---
#define QK_K 256

// --- Block definition for INT4 ---
#define Q_BLOCK_SIZE_INT4 32
typedef struct {
    float d;
    int8_t qs[Q_BLOCK_SIZE_INT4 / 2];
} block_int4;
static_assert(sizeof(block_int4) == sizeof(float) + Q_BLOCK_SIZE_INT4 / 2, "wrong int4 block size/padding");


// --- Block definition for Q4_0 ---
#define Q_BLOCK_SIZE_Q4_0 32
typedef struct {
    float d;
    uint8_t qs[Q_BLOCK_SIZE_Q4_0 / 2];
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + Q_BLOCK_SIZE_Q4_0 / 2, "wrong q4_0 block size/padding");

// --- Block definition for Q8_0 ---
#define Q_BLOCK_SIZE_Q8_0 32
typedef struct {
    float d;
    int8_t qs[Q_BLOCK_SIZE_Q8_0];
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(float) + Q_BLOCK_SIZE_Q8_0, "wrong q8_0 block size/padding");

// --- Block definition for Q2_K ---
typedef struct {
    float d;
    float dmin;
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
} block_q2_k;
static_assert(sizeof(block_q2_k) == sizeof(float) * 2 + QK_K / 16 + QK_K / 4, "wrong q2_k block size/padding");


// --- Function Prototypes ---
// Main entry points called by the Python bindings
std::vector<char> quantize_model(const float* B, ssize_t M, ssize_t K, const std::string& qtype);

void matmul_quantized_cpu(const void* B_quantized, const float* A, float* Y,
                          ssize_t N, ssize_t K, ssize_t M, const std::string& qtype);

// Forward declaration for Vulkan implementation
#ifdef HIERARCHOS_USE_VULKAN
void matmul_quantized_vulkan(const void* B_quantized, const float* A, float* Y,
                             ssize_t N, ssize_t K, ssize_t M, const std::string& qtype);
#endif


// New debug function prototype
std::vector<float> dequantize_row_cpp(const pybind11::bytes& B_quantized_row, const std::string& qtype, ssize_t K);
