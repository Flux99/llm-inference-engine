# Require CMake 3.20 or newer to support modern C++ features and target properties
cmake_minimum_required(VERSION 3.20)

# Define the project name and declare that we use C++ (CXX)
project(llm_inference_engine CXX)

# Enforce the modern C++20 standard (required for concepts, std::span, etc.)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json so IDEs (clangd, VSCode) have full code intelligence
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler optimization & warning flags for GCC and Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -Wall -Wextra -Wpedantic : Turn on strict compiler warnings to catch subtle bugs
    # -O3                      : Maximum optimization (loop unrolling, inlining, vectorization)
    # -mavx2 -mfma             : Emit AVX2 (256-bit SIMD) and Fused Multiply-Add CPU instructions
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -mavx2 -mfma)
endif()

# Add include/ and tests/ directories to the header search path
include_directories(include tests)

# Define engine_core as an INTERFACE (header-only) library
add_library(engine_core INTERFACE)

# Expose include/ to any target that links against engine_core
target_include_directories(engine_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the CLI executable from src/main.cpp
add_executable(engine_cli src/main.cpp)

# Link engine_cli with our engine_core headers
target_link_libraries(engine_cli PRIVATE engine_core)

# Enable CTest framework for automated test runs
enable_testing()

# Define the unit testing executable from tests/test_tensor.cpp
add_executable(engine_tests tests/test_tensor.cpp)

# Link engine_tests with engine_core headers
target_link_libraries(engine_tests PRIVATE engine_core)

# Register engine_tests with CTest under the test name "TensorUnitTests"
add_test(NAME TensorUnitTests COMMAND engine_tests)