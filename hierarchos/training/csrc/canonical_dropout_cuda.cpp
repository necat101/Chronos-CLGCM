#include <torch/extension.h>

#include <cstdint>

torch::Tensor canonical_dropout_cuda(
    torch::Tensor input,
    std::uint64_t seed,
    std::uint64_t start_word,
    std::uint64_t word_count,
    std::uint32_t threshold,
    double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "canonical_dropout",
        &canonical_dropout_cuda,
        "Hierarchos canonical Philox dropout (CUDA)");
}
