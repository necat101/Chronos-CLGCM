import numpy as np
import hierarchos_matmul
import time
import inspect

# ===================================================================
# Python "Golden" Reference Implementations
# ===================================================================

def dequantize_int4_golden(B_row_quantized):
    """Dequantizes a single row of INT4 data in Python."""
    block_size = 32
    bytes_per_block = 4 + 16
    num_blocks = len(B_row_quantized) // bytes_per_block
    B_row_dequantized = np.zeros(num_blocks * block_size, dtype=np.float32)

    for i in range(num_blocks):

        offset = i * bytes_per_block
        d = np.frombuffer(B_row_quantized[offset:offset+4], dtype=np.float32)[0]
        qs_packed = np.frombuffer(B_row_quantized[offset+4:offset+bytes_per_block], dtype=np.int8)

        for j in range(block_size // 2):
            packed_val = qs_packed[j]
            q1 = packed_val & 0x0F
            if q1 > 7: q1 -= 16
            q2 = (packed_val >> 4)
            B_row_dequantized[i*block_size + j*2]     = q1 * d
            B_row_dequantized[i*block_size + j*2 + 1] = q2 * d
            
    return B_row_dequantized

def dequantize_q4_0_golden(B_row_quantized):
    """Dequantizes a single row of Q4_0 data in Python."""
    block_size = 32
    bytes_per_block = 4 + 16
    num_blocks = len(B_row_quantized) // bytes_per_block
    B_row_dequantized = np.zeros(num_blocks * block_size, dtype=np.float32)

    for i in range(num_blocks):
        offset = i * bytes_per_block
        d = np.frombuffer(B_row_quantized[offset:offset+4], dtype=np.float32)[0]
        qs_packed = np.frombuffer(B_row_quantized[offset+4:offset+bytes_per_block], dtype=np.uint8)
        
        for j in range(block_size // 2):
            q1 = int(qs_packed[j] & 0x0F) - 8
            q2 = int(qs_packed[j] >> 4) - 8
            B_row_dequantized[i*block_size + j*2] = q1 * d
            B_row_dequantized[i*block_size + j*2 + 1] = q2 * d

    return B_row_dequantized

def dequantize_q8_0_golden(B_row_quantized):
    """Dequantizes a single row of Q8_0 data in Python."""
    block_size = 32
    bytes_per_block = 4 + 32
    num_blocks = len(B_row_quantized) // bytes_per_block
    B_row_dequantized = np.zeros(num_blocks * block_size, dtype=np.float32)

    for i in range(num_blocks):
        offset = i * bytes_per_block
        d = np.frombuffer(B_row_quantized[offset:offset+4], dtype=np.float32)[0]
        qs = np.frombuffer(B_row_quantized[offset+4:offset+bytes_per_block], dtype=np.int8)
        B_row_dequantized[i*block_size:(i+1)*block_size] = qs * d
        
    return B_row_dequantized

def dequantize_q2_k_golden(B_row_quantized):
    """Dequantizes a single row of Q2_K data in Python."""
    block_size = 256
    bytes_per_block = 4 + 4 + 16 + 64
    num_blocks = len(B_row_quantized) // bytes_per_block
    B_row_dequantized = np.zeros(num_blocks * block_size, dtype=np.float32)


    for i in range(num_blocks):
        offset = i * bytes_per_block
        d = np.frombuffer(B_row_quantized[offset:offset+4], dtype=np.float32)[0]
        dmin = np.frombuffer(B_row_quantized[offset+4:offset+8], dtype=np.float32)[0]
        scales = np.frombuffer(B_row_quantized[offset+8:offset+24], dtype=np.uint8)
        qs = np.frombuffer(B_row_quantized[offset+24:offset+bytes_per_block], dtype=np.uint8)
        
        for j in range(0, block_size, 32):
            scale_byte = scales[j//32]
            scale1 = ((scale_byte & 0x0F) / 15.0) * d
            scale2 = ((scale_byte >> 4) / 15.0) * d
            
            for l in range(16):
                qi = (qs[(j+l)//4] >> (2*((j+l)%4))) & 3
                B_row_dequantized[i*block_size + j+l] = qi * scale1 + dmin
            for l in range(16):
                qi = (qs[(j+16+l)//4] >> (2*((j+16+l)%4))) & 3
                B_row_dequantized[i*block_size + j+16+l] = qi * scale2 + dmin
    
    return B_row_dequantized

# ===================================================================
# NEW "TRUE" GOLDEN REFERENCE
# This function mimics the C++ scalar kernel's logic exactly.
# ===================================================================
def matmul_quantized_scalar_golden(A, B_quantized, M, qtype):
    """
    Performs quantized matmul by perfectly mimicking the C++ scalar fallback logic.
    This is the TRUE reference for verifying the SIMD kernels.
    """
    N, K = A.shape
    Y = np.zeros((N, M), dtype=np.float32)
    
    # Get block size and bytes per block based on qtype
    if qtype == "INT4" or qtype == "Q4_0" or qtype == "Q8_0":
        block_size_k = 32
    elif qtype == "Q2_K":
        block_size_k = 256
    else:
        raise ValueError("Unknown qtype")

    num_blocks_per_row = K // block_size_k
    bytes_per_block = len(B_quantized) // (M * num_blocks_per_row)
    bytes_per_row = num_blocks_per_row * bytes_per_block

    # Dequantize B row-by-row
    B_dequant = np.zeros((M, K), dtype=np.float32)
    dequant_func, _ = DEQUANT_MAP[qtype]
    for m in range(M):
        offset = m * bytes_per_row
        end_offset = (m + 1) * bytes_per_row
        B_row_quantized = B_quantized[offset:end_offset]
        B_dequant[m, :] = dequant_func(B_row_quantized)

    # Perform matmul with the same loop structure as the C++ kernel
    for n in range(N):
        for m in range(M):
            # This is a simple dot product, matching the C++ accumulation order
            Y[n, m] = np.dot(A[n, :], B_dequant[m, :])
            
    return Y

DEQUANT_MAP = {
    "INT4": (dequantize_int4_golden, 32),
    "Q4_0": (dequantize_q4_0_golden, 32),
    "Q8_0": (dequantize_q8_0_golden, 32),
    "Q2_K": (dequantize_q2_k_golden, 256),
}

# ===================================================================
# Debugging and Test Runner
# ===================================================================

def check_correctness(a, b, name="", threshold=0.1):
    """Compares two matrices and prints their error."""
    max_err = np.max(np.abs(a - b))
    mean_abs_err = np.mean(np.abs(a - b))
    print(f"[{name}]")

    print(f"  - Max Abs Error:  {max_err:.6f}")
    print(f"  - Mean Abs Error: {mean_abs_err:.6f}")
    if max_err > threshold:
        print(f"  - ⚠️ FAILURE: Max error is above the threshold of {threshold}!")
    else:
        print(f"  - ✅ SUCCESS: Accuracy is within the expected range.")
    return max_err > threshold

def run_deep_debug(N, K, M, qtype, threshold=0.1):
    dequant_func, block_size = DEQUANT_MAP[qtype]

    print(f"\nhierarchos Kernel Deep Debug Script: {qtype}")
    print("="*60)
    print(f"Matrix dimensions: A({N}, {K}), B({M}, {K}), Y({N}, {M})")
    print("-" * 60)

    np.random.seed(42)
    A = (np.random.rand(N, K).astype(np.float32) - 0.5) * 2
    B_float = (np.random.rand(M, K).astype(np.float32) - 0.5) * 2

    print("1. Quantizing matrix B using C++...")
    B_quantized_bytes = hierarchos_matmul.quantize(B_float, qtype=qtype)
    print("   Done.")

    print("\n2. Running Matmul Kernels and References...")
    # Reference: Pure NumPy with original float matrix
    Y_ref_fp32 = A @ B_float.T
    
    # C++ CPU Kernel (AVX, NEON, or Scalar)
    start_time_cpu = time.time()
    Y_cpu = hierarchos_matmul.matmul_quantized(A, B_quantized_bytes, M, qtype=qtype, device="cpu")
    end_time_cpu = time.time()
    print(f"   CPU Kernel Time: {(end_time_cpu - start_time_cpu) * 1000:.2f} ms")

    Y_vulkan = None
    if hierarchos_matmul.VULKAN_SUPPORT:
        try:
            # Vulkan GPU Kernel
            start_time_vk = time.time()
            Y_vulkan = hierarchos_matmul.matmul_quantized(A, B_quantized_bytes, M, qtype=qtype, device="vulkan")
            end_time_vk = time.time()
            print(f"   Vulkan Kernel Time: {(end_time_vk - start_time_vk) * 1000:.2f} ms")
        except RuntimeError as e:
            print(f"   Vulkan execution failed: {e}")
    else:
        print("   Vulkan support not compiled in this kernel.")

    print("   Done.")

    print("\n" + "="*60)
    print("Final Accuracy Report:")
    print("-" * 60)
    
    is_failing = False
    
    # TEST 1: Compare C++ CPU kernel against the original FP32 result.
    # This measures the error from quantization itself. The threshold is larger.
    print("--- C++ CPU vs. NumPy FP32 (Quantization Error Check) ---")
    is_failing |= check_correctness(Y_ref_fp32, Y_cpu, f"NumPy FP32 vs. C++ CPU {qtype}", threshold=threshold)
    print("-" * 60)

    # TEST 2: If Vulkan ran, compare its output to the CPU output.
    # They should be nearly identical.
    if Y_vulkan is not None:
        print("--- C++ Vulkan vs. C++ CPU (Backend Consistency Check) ---")
        # Threshold should be very low, accounting only for minor floating point differences.
        is_failing |= check_correctness(Y_cpu, Y_vulkan, f"C++ CPU vs. Vulkan {qtype}", threshold=0.01)
        print("-" * 60)
    
    if is_failing:
        print("🐞 Overall test status: FAILED")
    else:
        print("🎉 Overall test status: PASSED")
    print("="*60)

if __name__ == "__main__":
    run_deep_debug(N=1, K=4096, M=2048, qtype="INT4", threshold=5.0)
    run_deep_debug(N=1, K=4096, M=4096, qtype="Q4_0", threshold=6.0)
    run_deep_debug(N=1, K=4096, M=4096, qtype="Q8_0", threshold=0.5)
    run_deep_debug(N=1, K=4096, M=2048, qtype="Q2_K", threshold=25.0)
