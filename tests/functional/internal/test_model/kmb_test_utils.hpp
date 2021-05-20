//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/utils/IE/blob.hpp"

#include <inference_engine.hpp>
#include <vpu/utils/io.hpp>
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
    os << (method == CompareMethod::Absolute ? "CompareMethod::Absolute" :
           method == CompareMethod::Relative ? "CompareMethod::Relative" :
                                               "CompareMethod::Combined");
    return os;
}

void compareBlobs(
        const Blob::Ptr& actual,
        const Blob::Ptr& expected,
        float tolerance,
        CompareMethod method = CompareMethod::Absolute);

//
// Helper structs
//

template <typename T>
struct Vec2D final {
    T x;
    T y;
};
template <typename T>
std::ostream& operator<<(std::ostream& os, const Vec2D<T>& v) {
    vpu::formatPrint(os, "[x:%v, y:%v]", v.x, v.y);
    return os;
}

struct Pad2D final {
    ptrdiff_t left;
    ptrdiff_t right;
    ptrdiff_t top;
    ptrdiff_t bottom;
};
inline std::ostream& operator<<(std::ostream& os, const Pad2D& p) {
    vpu::formatPrint(os, "[left:%v, right:%v, top:%v, bottom:%v]", p.left, p.right, p.top, p.bottom);
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
            vpu::formatPrint(os, "Native");
            break;
        case KernelType::Ocl:
            vpu::formatPrint(os, "OCL");
            break;
        case KernelType::Cpp:
            vpu::formatPrint(os, "CPP");
            break;
    }

    return os;
}
