#include "hierarchos_matmul.h"
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace py = pybind11;


// ==========================================================================================
// Quantization Implementations (With High-Accuracy Improvements)
// ==========================================================================================

template<typename BlockType>
void quantize_row_q(const float* x, BlockType* y, ssize_t k);

template<>
void quantize_row_q<block_int4>(const float* x, block_int4* y, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_INT4;
    const int num_blocks = static_cast<int>(k / q_block_size);

    for (int i = 0; i < num_blocks; ++i) {
        const float* x_block = x + i * q_block_size;
        
        float max_abs = 0.0f;
        for (int j = 0; j < q_block_size; ++j) {
            if (std::fabs(x_block[j]) > max_abs) {
                max_abs = std::fabs(x_block[j]);
            }
        }

        const float d_initial = (max_abs == 0.0f) ? 1.0f : max_abs / 8.0f;
        const float id_initial = (d_initial == 0.0f) ? 0.0f : 1.0f / d_initial;

        int8_t temp_qs[q_block_size];
        for (int j = 0; j < q_block_size; ++j) {
            temp_qs[j] = static_cast<int8_t>(roundf(std::max(-8.0f, std::min(7.0f, x_block[j] * id_initial))));
        }

        double sum_xq = 0.0;
        double sum_qq = 0.0;
        for (int j = 0; j < q_block_size; ++j) {
            sum_xq += static_cast<double>(x_block[j]) * temp_qs[j];
            sum_qq += static_cast<double>(temp_qs[j]) * temp_qs[j];
        }

        y[i].d = (sum_qq > 1e-12) ? static_cast<float>(sum_xq / sum_qq) : d_initial;

        for (int j = 0; j < q_block_size; j += 2) {
            y[i].qs[j / 2] = (temp_qs[j] & 0x0F) | (temp_qs[j + 1] << 4);
        }
    }
}

template<>
void quantize_row_q<block_q4_0>(const float* x, block_q4_0* y, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q4_0;
    const int num_blocks = static_cast<int>(k / q_block_size);

    for (int i = 0; i < num_blocks; ++i) {
        const float* x_block = x + i * q_block_size;

        float max_abs = 0.0f;
        for (int j = 0; j < q_block_size; ++j) {
            if (std::fabs(x_block[j]) > max_abs) {
                max_abs = std::fabs(x_block[j]);
            }
        }
        
        const float d_initial = (max_abs == 0.0f) ? 1.0f : max_abs / 8.0f;
        const float id_initial = (d_initial == 0.0f) ? 0.0f : 1.0f / d_initial;


        int8_t temp_qs[q_block_size];
        for (int j = 0; j < q_block_size; ++j) {
            temp_qs[j] = static_cast<int8_t>(roundf(std::max(-8.0f, std::min(7.0f, x_block[j] * id_initial))));
        }

        double sum_xq = 0.0;
        double sum_qq = 0.0;
        for (int j = 0; j < q_block_size; ++j) {
            sum_xq += static_cast<double>(x_block[j]) * temp_qs[j];
            sum_qq += static_cast<double>(temp_qs[j]) * temp_qs[j];
        }

        y[i].d = (sum_qq > 1e-12) ? static_cast<float>(sum_xq / sum_qq) : d_initial;

        for (int j = 0; j < q_block_size; j += 2) {
            y[i].qs[j / 2] = ((temp_qs[j] + 8) & 0x0F) | (((temp_qs[j + 1] + 8) & 0x0F) << 4);
        }
    }
}

template<>
void quantize_row_q<block_q8_0>(const float* x, block_q8_0* y, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q8_0;
    const int num_blocks = static_cast<int>(k / q_block_size);

    for (int i = 0; i < num_blocks; ++i) {
        const float* x_block = x + i * q_block_size;

        float max_abs = 0.0f;
        for (int j = 0; j < q_block_size; ++j) {
            if (std::fabs(x_block[j]) > max_abs) {
                max_abs = std::fabs(x_block[j]);
            }
        }

        const float d_initial = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
        const float id_initial = (d_initial == 0.0f) ? 0.0f : 1.0f / d_initial;

        for (int j = 0; j < q_block_size; ++j) {
            y[i].qs[j] = static_cast<int8_t>(roundf(std::max(-127.0f, std::min(127.0f, x_block[j] * id_initial))));
        }
        
        double sum_xq = 0.0;
        double sum_qq = 0.0;
        for(int j=0; j<q_block_size; ++j) {
            sum_xq += static_cast<double>(x_block[j]) * y[i].qs[j];
            sum_qq += static_cast<double>(y[i].qs[j]) * y[i].qs[j];
        }

        y[i].d = (sum_qq > 1e-12) ? static_cast<float>(sum_xq / sum_qq) : d_initial;
    }
}

template<>
void quantize_row_q<block_q2_k>(const float* x, block_q2_k* y, ssize_t k) {
    const int q_block_size = QK_K;
    const int num_blocks = static_cast<int>(k / q_block_size);

    for (int i = 0; i < num_blocks; i++) {
        const float* x_block = x + i * q_block_size;
        block_q2_k* y_block = y + i;

        float min_val = x_block[0];
        float max_val = x_block[0];
        for (int j = 1; j < q_block_size; j++) {
            if (x_block[j] < min_val) min_val = x_block[j];
            if (x_block[j] > max_val) max_val = x_block[j];
        }

        const float d = (max_val - min_val);
        y_block->dmin = min_val;
        y_block->d = d;
        const float id = d != 0.f ? 1.0f / d : 0.0f;

        for (int j = 0; j < q_block_size; j += 32) {
            float max_rescaled_abs_1 = 1e-12f;
            for (int l = 0; l < 16; ++l) {
                float val = x_block[j + l] - min_val;
                if (std::fabs(val) > max_rescaled_abs_1) max_rescaled_abs_1 = std::fabs(val);
            }
            float max_rescaled_abs_2 = 1e-12f;
            for (int l = 0; l < 16; ++l) {
                float val = x_block[j + 16 + l] - min_val;
                if (std::fabs(val) > max_rescaled_abs_2) max_rescaled_abs_2 = std::fabs(val);
            }

            const float d1_original = max_rescaled_abs_1 / 3.0f;
            const float d2_original = max_rescaled_abs_2 / 3.0f;
            const uint8_t s1 = static_cast<uint8_t>(roundf(std::min(15.f, d1_original * id * 15.f)));
            const uint8_t s2 = static_cast<uint8_t>(roundf(std::min(15.f, d2_original * id * 15.f)));
            y_block->scales[j / 32] = s1 | (s2 << 4);

            const float d1_recon = ((s1) / 15.0f) * d;
            const float d2_recon = ((s2) / 15.0f) * d;
            const float id1_recon = d1_recon > 1e-12f ? 1.0f / d1_recon : 0.0f;
            const float id2_recon = d2_recon > 1e-12f ? 1.0f / d2_recon : 0.0f;

            for (int l = 0; l < 16; l += 4) {
                uint8_t q1 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+l+0] - min_val) * id1_recon))));
                uint8_t q2 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+l+1] - min_val) * id1_recon))));
                uint8_t q3 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+l+2] - min_val) * id1_recon))));
                uint8_t q4 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+l+3] - min_val) * id1_recon))));
                y_block->qs[(j + l) / 4] = q1 | (q2 << 2) | (q3 << 4) | (q4 << 6);
            }
            for (int l = 0; l < 16; l += 4) {
                uint8_t q1 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+16+l+0] - min_val) * id2_recon))));
                uint8_t q2 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+16+l+1] - min_val) * id2_recon))));
                uint8_t q3 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+16+l+2] - min_val) * id2_recon))));
                uint8_t q4 = static_cast<uint8_t>(roundf(std::max(0.0f, std::min(3.0f, (x_block[j+16+l+3] - min_val) * id2_recon))));
                y_block->qs[(j + 16 + l) / 4] = q1 | (q2 << 2) | (q3 << 4) | (q4 << 6);
            }
        }
    }
}


// ==========================================================================================
// Matrix Multiplication Implementations
// ==========================================================================================
template<typename BlockType>
float dot_product_q(const float* a, const BlockType* b, ssize_t k);

#if defined(HIERARCHOS_USE_AVX512)
// --- AVX-512 Implementations ---
template<>
float dot_product_q<block_int4>(const float* a, const block_int4* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_INT4;
    const int num_blocks = static_cast<int>(k / q_block_size);
    __m512 acc_f32 = _mm512_setzero_ps();

    const __m256i low_mask = _mm256_set1_epi16(0x000F);
    const __m256i high_mask = _mm256_set1_epi16(0x00F0);
    
    for (int i = 0; i < num_blocks; ++i) {
        const __m512 scale_vec = _mm512_set1_ps(b[i].d);
        const float* a_block = a + i * q_block_size;

        __m128i b_packed_128 = _mm_loadu_si128((const __m128i*)(b[i].qs));
        __m256i b_packed_256 = _mm256_cvtepi8_epi16(b_packed_128);
        
        __m256i b_low_nibbles  = _mm256_and_si256(b_packed_256, low_mask);
        __m256i b_high_masked  = _mm256_and_si256(b_packed_256, high_mask);
        __m256i b_high_nibbles = _mm256_srli_epi16(b_high_masked, 4);
        
        __m256i b_s16_evn = _mm256_srai_epi16(_mm256_slli_epi16(b_low_nibbles, 12), 12);
        __m256i b_s16_odd = _mm256_srai_epi16(_mm256_slli_epi16(b_high_nibbles, 12), 12);
        
        __m256i b_interleaved_lo = _mm256_unpacklo_epi16(b_s16_evn, b_s16_odd);
        __m256i b_interleaved_hi = _mm256_unpackhi_epi16(b_s16_evn, b_s16_odd);

        __m256i part0 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x20);
        __m256i part1 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x31);
        
        __m512 b_dequant_0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(part0)), scale_vec);
        __m512 b_dequant_1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(part1)), scale_vec);

        acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_block + 0),  b_dequant_0, acc_f32);
        acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_block + 16), b_dequant_1, acc_f32);
    }
    return _mm512_reduce_add_ps(acc_f32);
}

template<>
float dot_product_q<block_q4_0>(const float* a, const block_q4_0* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q4_0;
    const int num_blocks = static_cast<int>(k / q_block_size);
    __m512 acc_f32 = _mm512_setzero_ps();

    const __m256i low_mask = _mm256_set1_epi16(0x000F);
    const __m256i bias = _mm256_set1_epi16(8);

    for (int i = 0; i < num_blocks; ++i) {
        const __m512 scale_vec = _mm512_set1_ps(b[i].d);
        const float* a_block = a + i * q_block_size;

        __m128i b_packed_128 = _mm_loadu_si128((const __m128i*)(b[i].qs));
        __m256i b_packed_256 = _mm256_cvtepu8_epi16(b_packed_128);

        __m256i b_low_nibbles  = _mm256_and_si256(b_packed_256, low_mask);
        __m256i b_high_nibbles = _mm256_srli_epi16(b_packed_256, 4);

        __m256i b_s16_evn = _mm256_sub_epi16(b_low_nibbles, bias);
        __m256i b_s16_odd = _mm256_sub_epi16(b_high_nibbles, bias);

        __m256i b_interleaved_lo = _mm256_unpacklo_epi16(b_s16_evn, b_s16_odd);
        __m256i b_interleaved_hi = _mm256_unpackhi_epi16(b_s16_evn, b_s16_odd);
        __m256i part0 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x20);
        __m256i part1 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x31);

        __m512 b_dequant_0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(part0)), scale_vec);
        __m512 b_dequant_1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(part1)), scale_vec);

        acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_block + 0),  b_dequant_0, acc_f32);
        acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_block + 16), b_dequant_1, acc_f32);
    }
    return _mm512_reduce_add_ps(acc_f32);
}

template<>
float dot_product_q<block_q8_0>(const float* a, const block_q8_0* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q8_0;
    const int num_blocks = static_cast<int>(k / q_block_size);
    __m512 acc_f32 = _mm512_setzero_ps();

    for (int i = 0; i < num_blocks; ++i) {
        const __m512 d_v = _mm512_set1_ps(b[i].d);
        const float* a_block = a + i * q_block_size;

        __m256i q_s8 = _mm256_loadu_si256((const __m256i*)(b[i].qs));
        __m512i q_s16 = _mm512_cvtepi8_epi16(q_s8);
        
        __m256i q_s16_lo = _mm512_castsi512_si256(q_s16);
        __m512i q_s32_0 = _mm512_cvtepi16_epi32(q_s16_lo);
        __m512 dq_0 = _mm512_mul_ps(_mm512_cvtepi32_ps(q_s32_0), d_v);
        acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_block + 0), dq_0, acc_f32);
        
        // ✅ BUGFIX: Replaced problematic intrinsic with a more compatible one.
        __m256i q_s16_hi = _mm512_extracti32x8_epi32(q_s16, 1);
        __m512i q_s32_1 = _mm512_cvtepi16_epi32(q_s16_hi);
        __m512 dq_1 = _mm512_mul_ps(_mm512_cvtepi32_ps(q_s32_1), d_v);
        acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_block + 16), dq_1, acc_f32);
    }
    return _mm512_reduce_add_ps(acc_f32);
}

template<>
float dot_product_q<block_q2_k>(const float* a, const block_q2_k* b, ssize_t k) {
    const int num_blocks = static_cast<int>(k / QK_K);
    __m512 acc_f32 = _mm512_setzero_ps();
    const __m512 scale_norm = _mm512_set1_ps(1.0f / 15.0f);

    for (int i = 0; i < num_blocks; ++i) {
        const float* a_block = a + i * QK_K;
        const block_q2_k* b_block = b + i;

        const __m512 d_v = _mm512_set1_ps(b_block->d);
        const __m512 dmin_v = _mm512_set1_ps(b_block->dmin);

        for (int j = 0; j < QK_K / 32; ++j) {
            const uint8_t scale_byte = b_block->scales[j];
            const float* a_ptr = a_block + j * 32;
            const uint8_t* qs_ptr = b_block->qs + j * 8;
            
            const __m512 s1_v = _mm512_mul_ps(_mm512_set1_ps(static_cast<float>(scale_byte & 0x0F)), scale_norm);
            const __m512 s2_v = _mm512_mul_ps(_mm512_set1_ps(static_cast<float>(scale_byte >> 4)), scale_norm);
            const __m512 d1_v = _mm512_mul_ps(s1_v, d_v);
            const __m512 d2_v = _mm512_mul_ps(s2_v, d_v);

            alignas(64) int32_t q_tmp[32];
            for (int l = 0; l < 8; ++l) {
                uint8_t q_byte = qs_ptr[l];
                q_tmp[l * 4 + 0] = (q_byte >> 0) & 3;
                q_tmp[l * 4 + 1] = (q_byte >> 2) & 3;
                q_tmp[l * 4 + 2] = (q_byte >> 4) & 3;
                q_tmp[l * 4 + 3] = (q_byte >> 6) & 3;
            }

            __m512i q32_0 = _mm512_load_si512((const __m512i*)&q_tmp[0]);
            __m512 qf_0 = _mm512_cvtepi32_ps(q32_0);
            __m512 dq_0 = _mm512_fmadd_ps(qf_0, d1_v, dmin_v);
            acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_ptr + 0), dq_0, acc_f32);
            
            __m512i q32_1 = _mm512_load_si512((const __m512i*)&q_tmp[16]);
            __m512 qf_1 = _mm512_cvtepi32_ps(q32_1);
            __m512 dq_1 = _mm512_fmadd_ps(qf_1, d2_v, dmin_v);
            acc_f32 = _mm512_fmadd_ps(_mm512_loadu_ps(a_ptr + 16), dq_1, acc_f32);
        }
    }
    return _mm512_reduce_add_ps(acc_f32);
}

#elif defined(HIERARCHOS_USE_AVX2)
// --- AVX2 Implementations ---
template<>
float dot_product_q<block_int4>(const float* a, const block_int4* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_INT4;
    const int num_blocks = static_cast<int>(k / q_block_size);
    __m256 acc_f32 = _mm256_setzero_ps();
    
    const __m256i low_mask = _mm256_set1_epi16(0x000F);
    const __m256i high_mask = _mm256_set1_epi16(0x00F0);

    for (int i = 0; i < num_blocks; ++i) {
        const float scale = b[i].d;

        const __m256 scale_vec = _mm256_set1_ps(scale);
        
        __m128i b_packed_128 = _mm_loadu_si128((const __m128i*)(b[i].qs));
        __m256i b_packed_256 = _mm256_cvtepi8_epi16(b_packed_128);

        __m256i b_low_nibbles  = _mm256_and_si256(b_packed_256, low_mask);
        __m256i b_high_masked  = _mm256_and_si256(b_packed_256, high_mask);
        __m256i b_high_nibbles = _mm256_srli_epi16(b_high_masked, 4);
        
        __m256i b_s16_evn = _mm256_srai_epi16(_mm256_slli_epi16(b_low_nibbles, 12), 12);
        __m256i b_s16_odd = _mm256_srai_epi16(_mm256_slli_epi16(b_high_nibbles, 12), 12);
        
        __m256i b_interleaved_lo = _mm256_unpacklo_epi16(b_s16_evn, b_s16_odd);
        __m256i b_interleaved_hi = _mm256_unpackhi_epi16(b_s16_evn, b_s16_odd);

        __m256i part0 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x20);
        __m256i part1 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x31);

        __m256 b_dequant_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(part0))), scale_vec);
        __m256 b_dequant_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(part0, 1))), scale_vec);
        __m256 b_dequant_2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(part1))), scale_vec);
        __m256 b_dequant_3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(part1, 1))), scale_vec);

        const float* a_block = a + i * q_block_size;
        __m256 a_f32_0 = _mm256_loadu_ps(a_block);
        __m256 a_f32_1 = _mm256_loadu_ps(a_block + 8);
        __m256 a_f32_2 = _mm256_loadu_ps(a_block + 16);
        __m256 a_f32_3 = _mm256_loadu_ps(a_block + 24);

        acc_f32 = _mm256_fmadd_ps(a_f32_0, b_dequant_0, acc_f32);
        acc_f32 = _mm256_fmadd_ps(a_f32_1, b_dequant_1, acc_f32);
        acc_f32 = _mm256_fmadd_ps(a_f32_2, b_dequant_2, acc_f32);
        acc_f32 = _mm256_fmadd_ps(a_f32_3, b_dequant_3, acc_f32);
    }

    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(acc_f32, 1), _mm256_castps256_ps128(acc_f32));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

template<>
float dot_product_q<block_q4_0>(const float* a, const block_q4_0* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q4_0;
    const int num_blocks = static_cast<int>(k / q_block_size);
    __m256 acc = _mm256_setzero_ps();
    
    const __m256i low_mask = _mm256_set1_epi16(0x000F);
    const __m256i bias = _mm256_set1_epi16(8);

    for (int i = 0; i < num_blocks; ++i) {
        const __m256 d_v = _mm256_set1_ps(b[i].d);
        const float* a_block = a + i * q_block_size;
        
        __m128i b_packed_128 = _mm_loadu_si128((const __m128i*)(b[i].qs));
        __m256i b_packed_256 = _mm256_cvtepu8_epi16(b_packed_128);

        __m256i b_low_nibbles  = _mm256_and_si256(b_packed_256, low_mask);
        __m256i b_high_nibbles = _mm256_srli_epi16(b_packed_256, 4);

        __m256i b_s16_evn = _mm256_sub_epi16(b_low_nibbles, bias);
        __m256i b_s16_odd = _mm256_sub_epi16(b_high_nibbles, bias);

        __m256i b_interleaved_lo = _mm256_unpacklo_epi16(b_s16_evn, b_s16_odd);
        __m256i b_interleaved_hi = _mm256_unpackhi_epi16(b_s16_evn, b_s16_odd);
        __m256i part0 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x20);
        __m256i part1 = _mm256_permute2x128_si256(b_interleaved_lo, b_interleaved_hi, 0x31);

        __m256 b_dequant_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(part0))), d_v);
        __m256 b_dequant_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(part0, 1))), d_v);
        __m256 b_dequant_2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(part1))), d_v);
        __m256 b_dequant_3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(part1, 1))), d_v);

        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_block + 0),  b_dequant_0, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_block + 8),  b_dequant_1, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_block + 16), b_dequant_2, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_block + 24), b_dequant_3, acc);
    }

    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

template<>
float dot_product_q<block_q8_0>(const float* a, const block_q8_0* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q8_0;
    const int num_blocks = static_cast<int>(k / q_block_size);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < num_blocks; ++i) {
        const __m256 d_v = _mm256_set1_ps(b[i].d);
        const float* a_block = a + i * q_block_size;

        for (int j = 0; j < 2; ++j) {
            __m128i q_packed_128 = _mm_loadu_si128((const __m128i*)(b[i].qs + j * 16));
            __m256i q_16 = _mm256_cvtepi8_epi16(q_packed_128);

            __m128i q_16_lo = _mm256_castsi256_si128(q_16);
            __m256i q_32_0 = _mm256_cvtepi16_epi32(q_16_lo);
            __m256 dq_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_32_0), d_v);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_block + j * 16 + 0), dq_0, acc);

            __m128i q_16_hi = _mm256_extracti128_si256(q_16, 1);
            __m256i q_32_1 = _mm256_cvtepi16_epi32(q_16_hi);
            __m256 dq_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_32_1), d_v);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_block + j * 16 + 8), dq_1, acc);
        }
    }

    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

template<>
float dot_product_q<block_q2_k>(const float* a, const block_q2_k* b, ssize_t k) {
    const int num_blocks = static_cast<int>(k / QK_K);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < num_blocks; ++i) {
        const float* a_block = a + i * QK_K;
        const block_q2_k* b_block = b + i;

        const __m256 d_v = _mm256_set1_ps(b_block->d);
        const __m256 dmin_v = _mm256_set1_ps(b_block->dmin);
        const __m256 scale_norm = _mm256_set1_ps(1.0f / 15.0f);

        for (int j = 0; j < QK_K / 32; ++j) {
            const uint8_t scale_byte = b_block->scales[j];
            const float* a_ptr = a_block + j * 32;
            const uint8_t* qs_ptr = b_block->qs + j * 8;

            const __m256 s1_v = _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(scale_byte & 0x0F)), scale_norm);
            const __m256 s2_v = _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(scale_byte >> 4)), scale_norm);
            const __m256 d1_v = _mm256_mul_ps(s1_v, d_v);
            const __m256 d2_v = _mm256_mul_ps(s2_v, d_v);

            alignas(32) int32_t q_tmp[32];
            for (int l = 0; l < 8; ++l) {
                uint8_t q_byte = qs_ptr[l];
                q_tmp[l * 4 + 0] = (q_byte >> 0) & 3;
                q_tmp[l * 4 + 1] = (q_byte >> 2) & 3;
                q_tmp[l * 4 + 2] = (q_byte >> 4) & 3;
                q_tmp[l * 4 + 3] = (q_byte >> 6) & 3;
            }

            __m256i q32_0 = _mm256_load_si256((const __m256i*)&q_tmp[0]);
            __m256 qf_0 = _mm256_cvtepi32_ps(q32_0);
            __m256 dq_0 = _mm256_fmadd_ps(qf_0, d1_v, dmin_v);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr + 0), dq_0, acc);

            __m256i q32_1 = _mm256_load_si256((const __m256i*)&q_tmp[8]);
            __m256 qf_1 = _mm256_cvtepi32_ps(q32_1);
            __m256 dq_1 = _mm256_fmadd_ps(qf_1, d1_v, dmin_v);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr + 8), dq_1, acc);

            __m256i q32_2 = _mm256_load_si256((const __m256i*)&q_tmp[16]);
            __m256 qf_2 = _mm256_cvtepi32_ps(q32_2);
            __m256 dq_2 = _mm256_fmadd_ps(qf_2, d2_v, dmin_v);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr + 16), dq_2, acc);

            __m256i q32_3 = _mm256_load_si256((const __m256i*)&q_tmp[24]);
            __m256 qf_3 = _mm256_cvtepi32_ps(q32_3);
            __m256 dq_3 = _mm256_fmadd_ps(qf_3, d2_v, dmin_v);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr + 24), dq_3, acc);
        }
    }

    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif // HIERARCHOS_USE_AVX2 / AVX512

#if defined(HIERARCHOS_USE_NEON)
// --- NEON Implementations (Corrected & Refactored) ---

template<>
float dot_product_q<block_int4>(const float* a, const block_int4* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_INT4;
    const int num_blocks = static_cast<int>(k / q_block_size);
    float32x4_t acc_v = vdupq_n_f32(0.0f);

    const int8x8_t low_mask = vdup_n_s8(0x0F);

    for (int i = 0; i < num_blocks; ++i) {
        const float32x4_t d_v = vdupq_n_f32(b[i].d);
        const float* a_block = a + i * q_block_size;

        // Process 32 values in two chunks of 16
        for (int j = 0; j < 2; ++j) {
            const int8x8_t packed_s8 = vld1_s8(b[i].qs + j * 8);

            // Low nibbles: mask and then sign extend
            const int8x8_t low_nibbles = vand_s8(packed_s8, low_mask);
            const int8x8_t q_s8_lo = vshr_n_s8(vshl_n_s8(low_nibbles, 4), 4);

            // 🐞 BUGFIX: High nibbles must be shifted logically (unsigned) BEFORE sign extension.
            // The original arithmetic shift `vshr_n_s8` could be corrupted by the low nibble's bits.
            const int8x8_t high_nibbles = vreinterpret_s8_u8(vshr_n_u8(vreinterpret_u8_s8(packed_s8), 4));
            const int8x8_t q_s8_hi = vshr_n_s8(vshl_n_s8(high_nibbles, 4), 4);
            
            // Interleave low and high nibbles: [L0, H0, L1, H1, ...]
            const int8x8x2_t interleaved_8 = vzip_s8(q_s8_lo, q_s8_hi);
            const int8x16_t q_s8x16 = vcombine_s8(interleaved_8.val[0], interleaved_8.val[1]);

            // Widen and dequantize in two 8-element steps
            const int16x8_t q_s16_0 = vmovl_s8(vget_low_s8(q_s8x16));
            const int16x8_t q_s16_1 = vmovl_s8(vget_high_s8(q_s8x16));

            const float32x4_t dq_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_0))), d_v);
            const float32x4_t dq_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_0))), d_v);
            const float32x4_t dq_2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_1))), d_v);
            const float32x4_t dq_3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_1))), d_v);

            // Fused multiply-add
            const float* a_ptr = a_block + j * 16;
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 0), dq_0);
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 4), dq_1);
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 8), dq_2);
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 12), dq_3);
        }
    }
    return vaddvq_f32(acc_v);
}

template<>
float dot_product_q<block_q4_0>(const float* a, const block_q4_0* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q4_0;
    const int num_blocks = static_cast<int>(k / q_block_size);
    float32x4_t acc_v = vdupq_n_f32(0.0f);

    const uint8x8_t low_mask = vdup_n_u8(0x0F);
    const int8x8_t bias = vdup_n_s8(8);

    for (int i = 0; i < num_blocks; ++i) {
        const float32x4_t d_v = vdupq_n_f32(b[i].d);
        const float* a_block = a + i * q_block_size;

        for (int j = 0; j < 2; ++j) {
            const uint8x8_t packed_u8 = vld1_u8(b[i].qs + j * 8);

            const uint8x8_t low_nibbles = vand_u8(packed_u8, low_mask);
            const uint8x8_t high_nibbles = vshr_n_u8(packed_u8, 4);

            // Subtract bias to get signed values
            const int8x8_t q_s8_lo = vsub_s8(vreinterpret_s8_u8(low_nibbles), bias);
            const int8x8_t q_s8_hi = vsub_s8(vreinterpret_s8_u8(high_nibbles), bias);

            // Interleave low and high nibbles: [L0, H0, L1, H1, ...]
            const int8x8x2_t interleaved_8 = vzip_s8(q_s8_lo, q_s8_hi);
            const int8x16_t q_s8x16 = vcombine_s8(interleaved_8.val[0], interleaved_8.val[1]);
            
            // Widen and dequantize
            const int16x8_t q_s16_0 = vmovl_s8(vget_low_s8(q_s8x16));
            const int16x8_t q_s16_1 = vmovl_s8(vget_high_s8(q_s8x16));

            const float32x4_t dq_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_0))), d_v);
            const float32x4_t dq_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_0))), d_v);
            const float32x4_t dq_2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_1))), d_v);
            const float32x4_t dq_3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_1))), d_v);

            const float* a_ptr = a_block + j * 16;
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 0), dq_0);
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 4), dq_1);
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 8), dq_2);
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 12), dq_3);
        }
    }
    return vaddvq_f32(acc_v);
}

template<>
float dot_product_q<block_q8_0>(const float* a, const block_q8_0* b, ssize_t k) {
    const int q_block_size = Q_BLOCK_SIZE_Q8_0;
    const int num_blocks = static_cast<int>(k / q_block_size);
    float32x4_t acc_v = vdupq_n_f32(0.0f);

    for (int i = 0; i < num_blocks; ++i) {
        const float32x4_t d_v = vdupq_n_f32(b[i].d);
        const float* a_block = a + i * q_block_size;

        // 🐞 REFACTOR: The original logic was overly complex and likely had a copy-paste error.
        // This simpler implementation processes 8 elements at a time, which is much cleaner.
        for (int j = 0; j < 4; ++j) { // 4 chunks of 8 = 32 values
            const int8x8_t q_s8 = vld1_s8(b[i].qs + j * 8);

            const int16x8_t q_s16 = vmovl_s8(q_s8);

            // Widen to 32-bit integers
            const int32x4_t q_s32_lo = vmovl_s16(vget_low_s16(q_s16));
            const int32x4_t q_s32_hi = vmovl_s16(vget_high_s16(q_s16));
            
            // Convert to float and apply scale
            const float32x4_t dq_lo = vmulq_f32(vcvtq_f32_s32(q_s32_lo), d_v);
            const float32x4_t dq_hi = vmulq_f32(vcvtq_f32_s32(q_s32_hi), d_v);
            
            const float* a_ptr = a_block + j * 8;
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 0), dq_lo);
            acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 4), dq_hi);
        }
    }
    return vaddvq_f32(acc_v);
}

template<>
float dot_product_q<block_q2_k>(const float* a, const block_q2_k* b, ssize_t k) {
    const int num_blocks = static_cast<int>(k / QK_K);
    float32x4_t acc_v = vdupq_n_f32(0.0f);

    for (int i = 0; i < num_blocks; ++i) {
        const float* a_block = a + i * QK_K;
        const block_q2_k* b_block = b + i;
        const float32x4_t dmin_v = vdupq_n_f32(b_block->dmin);

        for (int j = 0; j < QK_K / 32; ++j) {
            const uint8_t scale_byte = b_block->scales[j];
            const float* a_ptr = a_block + j * 32;
            const uint8_t* qs_ptr = b_block->qs + j * 8;

            const float d1 = static_cast<float>(scale_byte & 0x0F) / 15.0f * b_block->d;
            const float d2 = static_cast<float>(scale_byte >> 4) / 15.0f * b_block->d;
            const float32x4_t d1_v = vdupq_n_f32(d1);
            const float32x4_t d2_v = vdupq_n_f32(d2);
            
            // Unpack 32 2-bit values into 32 32-bit integers
            alignas(16) int32_t q_tmp[32];
            for (int l = 0; l < 8; ++l) {
                uint8_t q_byte = qs_ptr[l];
                q_tmp[l * 4 + 0] = (q_byte >> 0) & 3;
                q_tmp[l * 4 + 1] = (q_byte >> 2) & 3;
                q_tmp[l * 4 + 2] = (q_byte >> 4) & 3;
                q_tmp[l * 4 + 3] = (q_byte >> 6) & 3;
            }

            // Process first 16 values with d1
            for (int l = 0; l < 4; ++l) {
                float32x4_t q_f32 = vcvtq_f32_s32(vld1q_s32(&q_tmp[l * 4]));
                float32x4_t dq = vmlaq_f32(dmin_v, q_f32, d1_v);
                acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + l * 4), dq);
            }

            // Process second 16 values with d2
            for (int l = 0; l < 4; ++l) {
                float32x4_t q_f32 = vcvtq_f32_s32(vld1q_s32(&q_tmp[16 + l * 4]));
                float32x4_t dq = vmlaq_f32(dmin_v, q_f32, d2_v);
                acc_v = vmlaq_f32(acc_v, vld1q_f32(a_ptr + 16 + l * 4), dq);
            }
        }
    }
    return vaddvq_f32(acc_v);
}
#endif // HIERARCHOS_USE_NEON


// Generic matmul loop that dispatches to the correct dot product
template<typename BlockType>
void matmul_q_cpu(const BlockType* B_quantized, const float* A, float* Y, ssize_t N, ssize_t K, ssize_t M) {
    #pragma omp parallel for schedule(dynamic)
    for (ssize_t n = 0; n < N; ++n) {
        const float* Arow = A + (size_t)n * K;
        for (ssize_t m = 0; m < M; ++m) {
            int block_size_k;
            if constexpr (std::is_same_v<BlockType, block_int4>) block_size_k = Q_BLOCK_SIZE_INT4;
            else if constexpr (std::is_same_v<BlockType, block_q2_k>) block_size_k = QK_K;
            else if constexpr (std::is_same_v<BlockType, block_q4_0>) block_size_k = Q_BLOCK_SIZE_Q4_0;
            else if constexpr (std::is_same_v<BlockType, block_q8_0>) block_size_k = Q_BLOCK_SIZE_Q8_0;

            const int num_blocks = static_cast<int>(K / block_size_k);
            const BlockType* Brow = B_quantized + (size_t)m * num_blocks;

#if defined(HIERARCHOS_USE_AVX512) || defined(HIERARCHOS_USE_AVX2) || defined(HIERARCHOS_USE_NEON)
            Y[(size_t)n * M + m] = dot_product_q(Arow, Brow, K);
#else
            // SCALAR FALLBACKS
            float sum = 0.0f;
            if constexpr (std::is_same_v<BlockType, block_int4>) {
                for (int i = 0; i < num_blocks; ++i) {
                    const float d = Brow[i].d;
                    for (int j = 0; j < Q_BLOCK_SIZE_INT4 / 2; ++j) {
                        const int8_t packed_qs = Brow[i].qs[j];
                        const int val1 = packed_qs & 0x0F;
                        const int q1 = val1 > 7 ? val1 - 16 : val1;
                        const int q2 = packed_qs >> 4;
                        sum += (Arow[i*Q_BLOCK_SIZE_INT4 + j*2 + 0] * q1) * d;
                        sum += (Arow[i*Q_BLOCK_SIZE_INT4 + j*2 + 1] * q2) * d;
                    }
                }
            } else if constexpr (std::is_same_v<BlockType, block_q8_0>) {
                for (int i = 0; i < num_blocks; ++i) {
                    const float d = Brow[i].d;
                    for (int j = 0; j < Q_BLOCK_SIZE_Q8_0; ++j) {
                        sum += (Arow[i*Q_BLOCK_SIZE_Q8_0 + j] * Brow[i].qs[j]) * d;
                    }
                }
            } else if constexpr (std::is_same_v<BlockType, block_q4_0>) {
                for (int i = 0; i < num_blocks; ++i) {
                    const float d = Brow[i].d;
                    for (int j = 0; j < Q_BLOCK_SIZE_Q4_0 / 2; ++j) {
                        const int q1 = (Brow[i].qs[j] & 0x0F) - 8;
                        const int q2 = (Brow[i].qs[j] >> 4)   - 8;
                        sum += (Arow[i*Q_BLOCK_SIZE_Q4_0 + j*2 + 0] * q1) * d;
                        sum += (Arow[i*Q_BLOCK_SIZE_Q4_0 + j*2 + 1] * q2) * d;
                    }
                }
            } else if constexpr (std::is_same_v<BlockType, block_q2_k>) {
                for (int i = 0; i < num_blocks; ++i) {
                    const float d = Brow[i].d;
                    const float dmin = Brow[i].dmin;
                    for (int j = 0; j < QK_K; j+=32) {
                        uint8_t scale_byte = Brow[i].scales[j/32];
                        float scale1 = ((scale_byte & 0x0F) / 15.0f) * d;
                        float scale2 = ((scale_byte >> 4)   / 15.0f) * d;

                        for(int l=0; l<16; ++l) {
                            int qi = (Brow[i].qs[(j+l)/4] >> (2*((j+l)%4))) & 3;
                            sum += (Arow[i*QK_K + j+l] * (qi * scale1 + dmin));
                        }
                        for(int l=0; l<16; ++l) {
                            int qi = (Brow[i].qs[(j+16+l)/4] >> (2*((j+16+l)%4))) & 3;
                            sum += (Arow[i*QK_K + j+16+l] * (qi * scale2 + dmin));
                        }
                    }
                }
            }
            Y[(size_t)n * M + m] = sum;
#endif
        }
    }
}


// ==========================================================================================
// Main Entry Points & Dequantization
// ==========================================================================================
std::vector<char> quantize_model(const float* B, ssize_t M, ssize_t K, const std::string& qtype) {
    std::vector<char> quantized_data;

    if (qtype == "INT4") {
        if (K % Q_BLOCK_SIZE_INT4 != 0) throw std::runtime_error("K must be a multiple of " + std::to_string(Q_BLOCK_SIZE_INT4));
        const size_t num_blocks_per_row = K / Q_BLOCK_SIZE_INT4;
        quantized_data.resize((size_t)M * num_blocks_per_row * sizeof(block_int4));
        auto* out = reinterpret_cast<block_int4*>(quantized_data.data());
        for (ssize_t m = 0; m < M; ++m) {
            quantize_row_q(B + m * K, out + m * num_blocks_per_row, K);
        }
    } else if (qtype == "Q4_0") {
        if (K % Q_BLOCK_SIZE_Q4_0 != 0) throw std::runtime_error("K must be a multiple of " + std::to_string(Q_BLOCK_SIZE_Q4_0));
        const size_t num_blocks_per_row = K / Q_BLOCK_SIZE_Q4_0;
        quantized_data.resize((size_t)M * num_blocks_per_row * sizeof(block_q4_0));
        auto* out = reinterpret_cast<block_q4_0*>(quantized_data.data());
        for (ssize_t m = 0; m < M; ++m) {
            quantize_row_q(B + m * K, out + m * num_blocks_per_row, K);
        }
    } else if (qtype == "Q8_0") {
        if (K % Q_BLOCK_SIZE_Q8_0 != 0) throw std::runtime_error("K must be a multiple of " + std::to_string(Q_BLOCK_SIZE_Q8_0));
        const size_t num_blocks_per_row = K / Q_BLOCK_SIZE_Q8_0;
        quantized_data.resize((size_t)M * num_blocks_per_row * sizeof(block_q8_0));
        auto* out = reinterpret_cast<block_q8_0*>(quantized_data.data());
        for (ssize_t m = 0; m < M; ++m) {
            quantize_row_q(B + m * K, out + m * num_blocks_per_row, K);
        }
    } else if (qtype == "Q2_K") {
        if (K % QK_K != 0) throw std::runtime_error("K must be a multiple of " + std::to_string(QK_K));
        const size_t num_blocks_per_row = K / QK_K;
        quantized_data.resize((size_t)M * num_blocks_per_row * sizeof(block_q2_k));
        auto* out = reinterpret_cast<block_q2_k*>(quantized_data.data());
        for (ssize_t m = 0; m < M; ++m) {
            quantize_row_q(B + m * K, out + m * num_blocks_per_row, K);
        }
    } else {
        throw std::runtime_error("Unsupported quantization type: " + qtype);
    }

    return quantized_data;
}

void matmul_quantized_cpu(const void* B_quantized, const float* A, float* Y,
                          ssize_t N, ssize_t K, ssize_t M, const std::string& qtype) {
    if (qtype == "INT4") {
        matmul_q_cpu(reinterpret_cast<const block_int4*>(B_quantized), A, Y, N, K, M);
    } else if (qtype == "Q4_0") {
        matmul_q_cpu(reinterpret_cast<const block_q4_0*>(B_quantized), A, Y, N, K, M);
    } else if (qtype == "Q8_0") {
        matmul_q_cpu(reinterpret_cast<const block_q8_0*>(B_quantized), A, Y, N, K, M);
    } else if (qtype == "Q2_K") {
        matmul_q_cpu(reinterpret_cast<const block_q2_k*>(B_quantized), A, Y, N, K, M);
    } else {
        throw std::runtime_error("Unsupported quantization type for matmul: " + qtype);
    }
}

template<typename BlockType>
void dequantize_block_q(const BlockType* block, float* out);

template<> void dequantize_block_q<block_int4>(const block_int4* block, float* out) {
    const float d = block->d;
    for (int j = 0; j < Q_BLOCK_SIZE_INT4 / 2; ++j) {
        const int8_t packed_qs = block->qs[j];
        const int val1 = packed_qs & 0x0F;
        out[j*2 + 0] = (val1 > 7 ? val1 - 16 : val1) * d;
        out[j*2 + 1] = (packed_qs >> 4) * d;
    }
}

template<> void dequantize_block_q<block_q4_0>(const block_q4_0* block, float* out) {
    const float d = block->d;
    for (int j = 0; j < Q_BLOCK_SIZE_Q4_0 / 2; ++j) {
        out[j*2 + 0] = ((block->qs[j] & 0x0F) - 8) * d;
        out[j*2 + 1] = ((block->qs[j] >> 4) - 8) * d;
    }
}

template<> void dequantize_block_q<block_q8_0>(const block_q8_0* block, float* out) {
    const float d = block->d;
    for (int j = 0; j < Q_BLOCK_SIZE_Q8_0; ++j) {
        out[j] = (block->qs[j] * d);
    }
}

template<> void dequantize_block_q<block_q2_k>(const block_q2_k* block, float* out) {
    const float d = block->d;
    const float dmin = block->dmin;
    for (int j = 0; j < QK_K; j+=32) {
        uint8_t scale_byte = block->scales[j/32];
        float scale1 = ((scale_byte & 0x0F) / 15.0f) * d;
        float scale2 = ((scale_byte >> 4) / 15.0f) * d;
        for(int l=0; l<16; ++l) {
            int qi = (block->qs[(j+l)/4] >> (2*((j+l)%4))) & 3;
            out[j+l] = qi * scale1 + dmin;
        }
        for(int l=0; l<16; ++l) {
            int qi = (block->qs[(j+16+l)/4] >> (2*((j+16+l)%4))) & 3;
            out[j+16+l] = qi * scale2 + dmin;
        }
    }
}

std::vector<float> dequantize_block_cpp(const py::bytes& B_quantized, const std::string& qtype, ssize_t block_idx) {
    const char *B_ptr = PyBytes_AsString(B_quantized.ptr());
    if (qtype == "INT4") {
        std::vector<float> out(Q_BLOCK_SIZE_INT4);
        const auto* blocks = reinterpret_cast<const block_int4*>(B_ptr);
        dequantize_block_q(&blocks[block_idx], out.data());
        return out;
    } else if (qtype == "Q4_0") {
        std::vector<float> out(Q_BLOCK_SIZE_Q4_0);
        const auto* blocks = reinterpret_cast<const block_q4_0*>(B_ptr);
        dequantize_block_q(&blocks[block_idx], out.data());
        return out;
    } else if (qtype == "Q8_0") {
        std::vector<float> out(Q_BLOCK_SIZE_Q8_0);
        const auto* blocks = reinterpret_cast<const block_q8_0*>(B_ptr);
        dequantize_block_q(&blocks[block_idx], out.data());
        return out;
    } else if (qtype == "Q2_K") {
        std::vector<float> out(QK_K);
        const auto* blocks = reinterpret_cast<const block_q2_k*>(B_ptr);
        dequantize_block_q(&blocks[block_idx], out.data());
        return out;
    }
    throw std::runtime_error("Unsupported qtype for dequantize_block");
}

std::vector<float> dequantize_row_cpp(const py::bytes& B_quantized_row, const std::string& qtype, ssize_t K) {
    const char *row_ptr = PyBytes_AsString(B_quantized_row.ptr());
    std::vector<float> out(K);

    if (qtype == "INT4") {
        const int num_blocks = static_cast<int>(K / Q_BLOCK_SIZE_INT4);
        const auto* blocks = reinterpret_cast<const block_int4*>(row_ptr);
        for (int i = 0; i < num_blocks; ++i) {
            dequantize_block_q(&blocks[i], out.data() + i * Q_BLOCK_SIZE_INT4);
        }
    } else if (qtype == "Q4_0") {
        const int num_blocks = static_cast<int>(K / Q_BLOCK_SIZE_Q4_0);
        const auto* blocks = reinterpret_cast<const block_q4_0*>(row_ptr);
        for (int i = 0; i < num_blocks; ++i) {
            dequantize_block_q(&blocks[i], out.data() + i * Q_BLOCK_SIZE_Q4_0);
        }
    } else if (qtype == "Q8_0") {
        const int num_blocks = static_cast<int>(K / Q_BLOCK_SIZE_Q8_0);
        const auto* blocks = reinterpret_cast<const block_q8_0*>(row_ptr);
        for (int i = 0; i < num_blocks; ++i) {
            dequantize_block_q(&blocks[i], out.data() + i * Q_BLOCK_SIZE_Q8_0);
        }
    } else if (qtype == "Q2_K") {
        const int num_blocks = static_cast<int>(K / QK_K);
        const auto* blocks = reinterpret_cast<const block_q2_k*>(row_ptr);
        for (int i = 0; i < num_blocks; ++i) {
            dequantize_block_q(&blocks[i], out.data() + i * QK_K);
        }
    } else {
        throw std::runtime_error("Unsupported qtype for dequantize_row");
    }
    return out;
}
