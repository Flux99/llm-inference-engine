#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace engine::core {

enum class DType {
    Float32,
    Float16,
    Int8,
    Int32
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int8:    return 1;
        case DType::Int32:   return 4;
        default:
            throw std::invalid_argument("Unknown DType");
    }
}

inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "Float32";
        case DType::Float16: return "Float16";
        case DType::Int8:    return "Int8";
        case DType::Int32:   return "Int32";
        default:             return "Unknown";
    }
}

// Compile-time helper mapping C++ types to DType enum
template <typename T>
struct TypeToDType;

template <> struct TypeToDType<float>   { static constexpr DType value = DType::Float32; };
template <> struct TypeToDType<int8_t>  { static constexpr DType value = DType::Int8; };
template <> struct TypeToDType<int32_t> { static constexpr DType value = DType::Int32; };

} // namespace engine::core
