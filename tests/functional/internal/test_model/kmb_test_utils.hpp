//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/core/format.hpp"

#include <inference_engine.hpp>
#include <ngraph/type/element_type.hpp>

#include <blob_factory.hpp>
#include <cstddef>
#include <random>

using namespace InferenceEngine;

//
// Blob utility
//

void fillUniform(const Blob::Ptr& blob, std::default_random_engine& rd, float min, float max);
void fillUniform(const Blob::Ptr& blob, std::default_random_engine& rd, int min, int max);

void fillNormal(const Blob::Ptr& blob, std::default_random_engine& rd, float mean, float stddev);

Blob::Ptr genBlobUniform(const TensorDesc& desc, std::default_random_engine& rd, float min, float max);
Blob::Ptr genBlobUniform(const TensorDesc& desc, std::default_random_engine& rd, int min, int max);

Blob::Ptr genBlobNormal(const TensorDesc& desc, std::default_random_engine& rd, float mean, float stddev);

template <typename T>
Blob::Ptr genBlobFromData(const TensorDesc& desc, const std::vector<T>& values) {
    auto blob = make_blob_with_precision(desc);
    blob->allocate();

    const auto outPtr = blob->buffer().as<T*>();
    IE_ASSERT(outPtr != nullptr);

    const auto& dims = desc.getDims();
    const auto dataSize = std::accumulate(begin(dims), end(dims), 1ul, std::multiplies<>{});
    IE_ASSERT(dataSize == values.size());

    std::copy_n(values.data(), values.size(), outPtr);

    return blob;
}

enum class CompareMethod { Absolute, Relative, Combined };

inline std::ostream& operator<<(std::ostream& os, CompareMethod method) {
    os << (method == CompareMethod::Absolute
                   ? "CompareMethod::Absolute"
                   : method == CompareMethod::Relative ? "CompareMethod::Relative" : "CompareMethod::Combined");
    return os;
}

void compareBlobs(const Blob::Ptr& actual, const Blob::Ptr& expected, float tolerance,
                  CompareMethod method = CompareMethod::Absolute);

//
// Helper structs
//

template <typename T>
struct Vec2D final {
    T x;
    T y;
};

template <typename Stream, typename T>
Stream& operator<<(Stream& os, const Vec2D<T>& v) {
    vpux::printTo(os, "[x:{0}, y:{1}]", v.x, v.y);
    return os;
}

struct Pad2D final {
    ptrdiff_t left;
    ptrdiff_t right;
    ptrdiff_t top;
    ptrdiff_t bottom;
};

template <typename Stream>
inline Stream& operator<<(Stream& os, const Pad2D& p) {
    vpux::printTo(os, "[left:{0}, right:{1}, top:{2}, bottom:{3}]", p.left, p.right, p.top, p.bottom);
    return os;
}

//
// Precision conversion functions
//

ngraph::element::Type precisionToType(const Precision& precision);

Precision typeToPrecision(const ngraph::element::Type& type);

//
// Custom layers
//

enum class KernelType : int {
    Native,
    Ocl,
    Cpp,
};

inline std::ostream& operator<<(std::ostream& os, const KernelType& p) {
    switch (p) {
    case KernelType::Native:
        vpux::printTo(os, "Native");
        break;
    case KernelType::Ocl:
        vpux::printTo(os, "OCL");
        break;
    case KernelType::Cpp:
        vpux::printTo(os, "CPP");
        break;
    }

    return os;
}
