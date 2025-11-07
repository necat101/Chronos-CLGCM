cmake_minimum_required(VERSION 3.16)
project(hierarchos_matmul LANGUAGES CXX)

option(FORCE_SCALAR "Force compilation without any SIMD intrinsics" OFF)
option(HIERARCHOS_BUILD_VULKAN "Enable Vulkan support" OFF)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

find_package(OpenMP)

if(MSVC)
    string(REPLACE "/RTC1" "/O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Od" "/O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "Flags used by the C++ compiler during release builds." FORCE)
endif()

# --- START Vulkan Optional Build ---
if(HIERARCHOS_BUILD_VULKAN)
    message(STATUS "Vulkan build requested. Searching for components...")
    find_package(Vulkan)
    if (Vulkan_FOUND)
        message(STATUS "Vulkan libraries found.")
        
        # --- Find glslangValidator first, then glslc ---
        find_program(GLSLC_COMPILER NAMES glslangValidator glslc)

        if (NOT GLSLC_COMPILER)
            message(WARNING "Vulkan shader compiler (glslangValidator or glslc) not found. Vulkan support disabled.")
            message(WARNING "Please install 'glslang-tools' or the Vulkan SDK and ensure the compiler is in your PATH.")
            set(Vulkan_FOUND FALSE)
            set(HIERARCHOS_BUILD_VULKAN FALSE)
        else()
            message(STATUS "Vulkan shader compiler found: ${GLSLC_COMPILER}")
        endif()
    else()
        message(WARNING "Vulkan libraries not found. Vulkan support disabled. Please install 'libvulkan-dev' or the Vulkan SDK.")
        set(HIERARCHOS_BUILD_VULKAN FALSE)
    endif()
endif()

if(HIERARCHOS_BUILD_VULKAN AND Vulkan_FOUND)
    message(STATUS "Vulkan support is ENABLED.")
    set(HIERARCHOS_USE_VULKAN TRUE)
    file(GLOB SHADER_SOURCES "shaders/*.comp")

    # Create the output directory for the compiled shaders
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")

    # --- ❗️❗️❗️ BEGIN FIX ❗️❗️❗️ ---
    # Conditionally add the -V flag based on the compiler's name
    get_filename_component(COMPILER_NAME ${GLSLC_COMPILER} NAME) # Gets "glslangValidator" or "glslc.exe" etc.
    set(GLSLC_ARGS "")
    if(COMPILER_NAME MATCHES "glslangValidator") # Check if the name *contains* glslangValidator
        message(STATUS "Using glslangValidator. Adding -V flag.")
        set(GLSLC_ARGS "-V")
    else()
        message(STATUS "Using glslc. Not adding -V flag.")
    endif()
    # --- ❗️❗️❗️ END FIX ❗️❗️❗️ ---

    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
        set(SHADER_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${SHADER_SPIRV}
            # --- Use the conditional args variable ---
            COMMAND ${GLSLC_COMPILER} ${GLSLC_ARGS} -o ${SHADER_SPIRV} ${SHADER_SOURCE}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling ${SHADER_NAME}.comp to SPIR-V using ${GLSLC_COMPILER}"
        )
        list(APPEND SHADER_SPIRV_FILES ${SHADER_SPIRV})
    endforeach()

    add_custom_target(Shaders ALL DEPENDS ${SHADER_SPIRV_FILES})
else()
    message(STATUS "Vulkan support is DISABLED.")
endif()
# --- END Vulkan Optional Build ---


pybind11_add_module(hierarchos_matmul
    hierarchos_matmul.cpp
    bindings.cpp
)

if(HIERARCHOS_BUILD_VULKAN AND Vulkan_FOUND)
    target_sources(hierarchos_matmul PRIVATE hierarchos_vulkan.cpp)
    target_link_libraries(hierarchos_matmul PRIVATE Vulkan::Vulkan)
    target_compile_definitions(hierarchos_matmul PRIVATE HIERARCHOS_USE_VULKAN)
    target_include_directories(hierarchos_matmul PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    add_dependencies(hierarchos_matmul Shaders)

    add_custom_command(
        TARGET hierarchos_matmul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_BINARY_DIR}/shaders"
                "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/shaders"
        COMMENT "Copying compiled shaders to output directory"
    )
endif()

target_compile_features(hierarchos_matmul PRIVATE cxx_std_17)

# --- START OF MODIFICATION: Robust SIMD Detection ---
include(CheckCXXSourceRuns)

if(NOT FORCE_SCALAR)
    if(MSVC)
        # MSVC uses /arch, which is safer as it doesn't require hardware checks.
        # We assume the user has Visual Studio configured correctly.
        target_compile_options(hierarchos_matmul PRIVATE /W3 /EHsc)
        message(STATUS "MSVC detected. Using /arch:AVX2 as a robust default.")
        target_compile_options(hierarchos_matmul PRIVATE /arch:AVX2)

    else() # For GCC/Clang
        target_compile_options(hierarchos_matmul PRIVATE -O3 -fPIC -std=c++17 -Wno-unused-variable)
        
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8-a")
            message(STATUS "ARM aarch64 architecture detected. Enabling NEON with -march=native.")
            target_compile_options(hierarchos_matmul PRIVATE -march=native)
        else()
            # Hierarchical check: Test for the best instruction set first.
            set(SIMD_FLAGS_FOUND FALSE)

            # 1. Check for AVX512
            file(WRITE "${CMAKE_BINARY_DIR}/check_avx512.cpp"
                "#include <immintrin.h>\n
                int main() {
                    __m512 a = _mm512_setzero_ps();
                    float f[16];
                    _mm512_storeu_ps(f, a);
                    return 0;
                }")
            set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw")
            check_cxx_source_runs("${CMAKE_BINARY_DIR}/check_avx512.cpp" HOST_SUPPORTS_AVX512)
            unset(CMAKE_REQUIRED_FLAGS)

            if(HOST_SUPPORTS_AVX512)
                message(STATUS "Host CPU supports AVX-512. Compiling with AVX-512 flags.")
                target_compile_options(hierarchos_matmul PRIVATE -mavx512f -mavx512bw -mfma)
                set(SIMD_FLAGS_FOUND TRUE)
            endif()

            # 2. If not AVX512, check for AVX2
            if(NOT SIMD_FLAGS_FOUND)
                file(WRITE "${CMAKE_BINARY_DIR}/check_avx2.cpp"
                    "#include <immintrin.h>\n
                    int main() {
                        __m256 a = _mm256_setzero_ps();
                        float f[8];
                        _mm256_storeu_ps(f, a);
                        return 0;
                    }")
                set(CMAKE_REQUIRED_FLAGS "-mavx2")
                check_cxx_source_runs("${CMAKE_BINARY_DIR}/check_avx2.cpp" HOST_SUPPORTS_AVX2)
                unset(CMAKE_REQUIRED_FLAGS)

                if(HOST_SUPPORTS_AVX2)
                    message(STATUS "Host CPU supports AVX2. Compiling with AVX2 flags.")
                    target_compile_options(hierarchos_matmul PRIVATE -mavx2 -mfma)
                    set(SIMD_FLAGS_FOUND TRUE)
                endif()
            endif()

            if(NOT SIMD_FLAGS_FOUND)
                message(WARNING "No AVX512 or AVX2 support detected on host. Building scalar version.")
            endif()
        endif()
    endif()
endif()
# --- END OF MODIFICATION ---


if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP support detected. Enabling parallelization.")
    target_link_libraries(hierarchos_matmul PRIVATE OpenMP::OpenMP_CXX)
endif()

# The post-build command to copy the final library to the project root
# This is handled by your setup.py now, but keeping it can be useful for direct cmake builds.
add_custom_command(
    TARGET hierarchos_matmul POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:hierarchos_matmul>"
        "${PROJECT_ROOT_DIR}/$<TARGET_FILE_NAME:hierarchos_matmul>"
    COMMENT "Copying kernel to project root..."
)
