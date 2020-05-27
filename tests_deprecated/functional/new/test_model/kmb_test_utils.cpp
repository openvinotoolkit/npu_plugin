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

#include "kmb_test_utils.hpp"

#include <precision_utils.h>
#include <blob_factory.hpp>
#include <blob_transform.hpp>
#include <single_layer_common.hpp>

#include <vpu/utils/checked_cast.hpp>

namespace vpu {

template <typename OutT, typename InT>
typename std::enable_if<
        std::is_floating_point<OutT>::value && std::is_integral<InT>::value,
    OutT>::type checked_cast(InT value) {
    // TODO: check range
    // TODO: move to DLDT repository
    return static_cast<OutT>(value);
}

}  // namespace vpu

namespace {

template <typename T, class Distribution>
void fill_(const Blob::Ptr& blob, std::default_random_engine& rd, Distribution& dist) {
    IE_ASSERT(blob != nullptr);

    const auto outPtr = blob->buffer().as<T*>();
    IE_ASSERT(outPtr != nullptr);

    std::generate_n(outPtr, blob->size(), [&]() {
        return vpu::checked_cast<T>(dist(rd));
    });
}
template <typename T, class Distribution, class ConvertOp>
void fill_(const Blob::Ptr& blob, std::default_random_engine& rd, Distribution& dist, const ConvertOp& op) {
    IE_ASSERT(blob != nullptr);

    const auto outPtr = blob->buffer().as<T*>();
    IE_ASSERT(outPtr != nullptr);

    std::generate_n(outPtr, blob->size(), [&]() {
        return op(dist(rd));
    });
}

template <typename T>
void fillUniform_(const Blob::Ptr& blob, std::default_random_engine& rd, T min, T max) {
    IE_ASSERT(blob != nullptr);

    switch (blob->getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        std::uniform_real_distribution<float> dist(vpu::checked_cast<float>(min), vpu::checked_cast<float>(max));
        fill_<float>(blob, rd, dist);
        break;
    }
    case Precision::FP16: {
        std::uniform_real_distribution<float> dist(vpu::checked_cast<float>(min), vpu::checked_cast<float>(max));
        fill_<ie_fp16>(blob, rd, dist, PrecisionUtils::f32tof16);
        break;
    }
    case Precision::I32: {
        std::uniform_int_distribution<int32_t> dist(vpu::checked_cast<int32_t>(min), vpu::checked_cast<int32_t>(max));
        fill_<int32_t>(blob, rd, dist);
        break;
    }
    case Precision::U8: {
        std::uniform_int_distribution<uint8_t> dist(vpu::checked_cast<uint8_t>(min), vpu::checked_cast<uint8_t>(max));
        fill_<uint8_t>(blob, rd, dist);
        break;
    }
    case Precision::I8: {
        std::uniform_int_distribution<int8_t> dist(vpu::checked_cast<int8_t>(min), vpu::checked_cast<int8_t>(max));
        fill_<int8_t>(blob, rd, dist);
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << blob->getTensorDesc().getPrecision();
    }
}

}  // namespace

void fillUniform(const Blob::Ptr& blob, std::default_random_engine& rd, float min, float max) {
    fillUniform_(blob, rd, min, max);
}

void fillUniform(const Blob::Ptr& blob, std::default_random_engine& rd, int min, int max) {
    fillUniform_(blob, rd, min, max);
}

void fillNormal(const Blob::Ptr& blob, std::default_random_engine& rd, float mean, float stddev) {
    IE_ASSERT(blob != nullptr);

    std::normal_distribution<float> dist(mean, stddev);

    switch (blob->getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        fill_<float>(blob, rd, dist);
        break;
    }
    case Precision::FP16: {
        fill_<ie_fp16>(blob, rd, dist, PrecisionUtils::f32tof16);
        break;
    }
    case Precision::I32: {
        fill_<int32_t>(blob, rd, dist);
        break;
    }
    case Precision::U8: {
        fill_<uint8_t>(blob, rd, dist);
        break;
    }
    case Precision::I8: {
        fill_<int8_t>(blob, rd, dist);
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << blob->getTensorDesc().getPrecision();
    }
}

namespace {

template <typename T>
Blob::Ptr genBlobUniform_(const TensorDesc& desc, std::default_random_engine& rd, T min, T max) {
    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    fillUniform(blob, rd, min, max);

    return blob;
}

}  // namespace

Blob::Ptr genBlobUniform(const TensorDesc& desc, std::default_random_engine& rd, float min, float max) {
    return genBlobUniform_(desc, rd, min, max);
}

Blob::Ptr genBlobUniform(const TensorDesc& desc, std::default_random_engine& rd, int min, int max) {
    return genBlobUniform_(desc, rd, min, max);
}

Blob::Ptr genBlobNormal(const TensorDesc& desc, std::default_random_engine& rd, float mean, float stddev) {
    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    fillNormal(blob, rd, mean, stddev);

    return blob;
}

void compareBlobs(const Blob::Ptr& actual, const Blob::Ptr& expected, float tolerance, CompareMethod method) {
    const auto actualFP32 = toFP32(actual);
    const auto expectedFP32 = toFP32(expected);

    switch (method) {
    case CompareMethod::Absolute:
        CompareCommonAbsolute(actualFP32, expectedFP32, tolerance);
        break;
    case CompareMethod::Relative:
        CompareCommonRelative(actualFP32, expectedFP32, tolerance);
        break;
    case CompareMethod::Combined:
        CompareCommonCombined(actualFP32, expectedFP32, tolerance);
        break;
    }
}

ngraph::element::Type precisionToType(const Precision& precision) {
    switch (precision) {
    case Precision::FP32:
        return ngraph::element::f32;
    case Precision::FP16:
        return ngraph::element::f16;
    case Precision::I64:
        return ngraph::element::i64;
    case Precision::I32:
        return ngraph::element::i32;
    case Precision::U8:
        return ngraph::element::u8;
    case Precision::I8:
        return ngraph::element::i8;
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << precision;
    }
}

Precision typeToPrecision(const ngraph::element::Type& type) {
    if (type == ngraph::element::f32) {
        return Precision::FP32;
    } else if (type == ngraph::element::f16) {
        return Precision::FP16;
    } else if (type == ngraph::element::i64) {
        return Precision::I64;
    } else if (type == ngraph::element::i32) {
        return Precision::I32;
    } else if (type == ngraph::element::u8) {
        return Precision::U8;
    } else if (type == ngraph::element::i8) {
        return Precision::I8;
    } else {
        THROW_IE_EXCEPTION << "Unsupported type " << type;
    }
}