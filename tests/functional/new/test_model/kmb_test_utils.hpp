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

#include <inference_engine.hpp>
#include <random>
#include <vpu/utils/io.hpp>
#include <cstddef>

using namespace InferenceEngine;

//
// Blob utility
//

void fillUniform(const Blob::Ptr& blob, std::default_random_engine& rd, float min, float max);
void fillNormal(const Blob::Ptr& blob, std::default_random_engine& rd, float mean, float stddev);

Blob::Ptr genBlobUniform(const TensorDesc& desc, std::default_random_engine& rd, float min, float max);
Blob::Ptr genBlobNormal(const TensorDesc& desc, std::default_random_engine& rd, float mean, float stddev);

Blob::Ptr makeScalarBlob(float val, const Precision& precision = Precision::FP32, size_t numDims = 1);

Blob::Ptr toFP32(const Blob::Ptr& in);
Blob::Ptr toFP16(const Blob::Ptr& in);

Blob::Ptr toDefLayout(const Blob::Ptr& in);

enum class CompareMethod { Absolute, Relative, Combined };

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
