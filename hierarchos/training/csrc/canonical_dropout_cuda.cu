#include <torch/extension.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85u;

struct Philox4x32 {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t w;
};

__device__ __forceinline__ Philox4x32 philox4x32_10(
    Philox4x32 counter,
    std::uint32_t key0,
    std::uint32_t key1) {
    #pragma unroll
    for (int round_index = 0; round_index < 10; ++round_index) {
        const std::uint64_t p0 =
            static_cast<std::uint64_t>(kPhiloxM0) * counter.x;
        const std::uint64_t p1 =
            static_cast<std::uint64_t>(kPhiloxM1) * counter.z;
        counter = {
            static_cast<std::uint32_t>(p1 >> 32) ^ counter.y ^ key0,
            static_cast<std::uint32_t>(p1),
            static_cast<std::uint32_t>(p0 >> 32) ^ counter.w ^ key1,
            static_cast<std::uint32_t>(p0),
        };
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

__device__ __forceinline__ std::uint32_t canonical_word(
    std::uint64_t seed,
    std::uint64_t word_offset) {
    const std::uint64_t block = word_offset >> 2;
    const unsigned int lane = static_cast<unsigned int>(word_offset & 3u);
    const Philox4x32 words = philox4x32_10(
        {
            static_cast<std::uint32_t>(block),
            static_cast<std::uint32_t>(block >> 32),
            0u,
            0u,
        },
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32));
    if (lane == 0u) {
        return words.x;
    }
    if (lane == 1u) {
        return words.y;
    }
    if (lane == 2u) {
        return words.z;
    }
    return words.w;
}

template <typename scalar_t>
__global__ void canonical_dropout_kernel(
    const scalar_t* input,
    scalar_t* output,
    std::uint64_t element_count,
    std::uint64_t seed,
    std::uint64_t start_word,
    std::uint32_t threshold,
    double scale) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const std::uint32_t random_word = canonical_word(seed, start_word + index);
    using acc_t = at::acc_type<scalar_t, true>;
    const acc_t scaled = static_cast<acc_t>(input[index]) * static_cast<acc_t>(scale);
    output[index] = random_word < threshold ? scalar_t(0) : static_cast<scalar_t>(scaled);
}

}  // namespace

torch::Tensor canonical_dropout_cuda(
    torch::Tensor input,
    std::uint64_t seed,
    std::uint64_t start_word,
    std::uint64_t word_count,
    std::uint32_t threshold,
    double scale) {
    TORCH_CHECK(input.is_cuda(), "canonical dropout input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "canonical dropout input must be contiguous");
    TORCH_CHECK(input.is_floating_point(), "canonical dropout input must be floating point");
    TORCH_CHECK(std::isfinite(scale) && scale >= 1.0, "canonical dropout scale must be finite and >= 1");

    const std::uint64_t element_count = static_cast<std::uint64_t>(input.numel());
    TORCH_CHECK(
        word_count >= element_count,
        "canonical dropout reservation has fewer words than the input tensor");
    if (element_count > 0) {
        TORCH_CHECK(
            start_word <= std::numeric_limits<std::uint64_t>::max() - (element_count - 1),
            "canonical dropout reservation word offset overflow");
    }

    auto output = torch::empty_like(input);
    if (element_count == 0) {
        return output;
    }

    constexpr unsigned int threads = 256;
    const std::uint64_t block_count_u64 = (element_count + threads - 1) / threads;
    TORCH_CHECK(
        block_count_u64 <= std::numeric_limits<unsigned int>::max(),
        "canonical dropout tensor is too large for one-dimensional CUDA dispatch");

    at::cuda::CUDAGuard device_guard(input.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    const dim3 blocks(static_cast<unsigned int>(block_count_u64));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "hierarchos_canonical_dropout_cuda",
        [&] {
            canonical_dropout_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                element_count,
                seed,
                start_word,
                threshold,
                scale);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
