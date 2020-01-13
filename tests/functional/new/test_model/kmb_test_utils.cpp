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

namespace {

template <typename T, class Distribution>
void fill_(const Blob::Ptr& blob, std::default_random_engine& rd, Distribution& dist) {
    IE_ASSERT(blob != nullptr);

    const auto outPtr = blob->buffer().as<T*>();
    IE_ASSERT(outPtr != nullptr);

    std::generate_n(outPtr, blob->size(), [&]() {
        return static_cast<T>(dist(rd));
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

}  // namespace

void fillUniform(const Blob::Ptr& blob, std::default_random_engine& rd, float min, float max) {
    IE_ASSERT(blob != nullptr);

    switch (blob->getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        std::uniform_real_distribution<float> dist(min, max);
        fill_<float>(blob, rd, dist);
        break;
    }
    case Precision::FP16: {
        std::uniform_real_distribution<float> dist(min, max);
        fill_<ie_fp16>(blob, rd, dist, PrecisionUtils::f32tof16);
        break;
    }
    case Precision::I32: {
        std::uniform_int_distribution<int32_t> dist(static_cast<int32_t>(min), static_cast<int32_t>(max));
        fill_<int32_t>(blob, rd, dist);
        break;
    }
    case Precision::U8: {
        std::uniform_int_distribution<uint8_t> dist(static_cast<uint8_t>(max), static_cast<uint8_t>(min));
        fill_<uint8_t>(blob, rd, dist);
        break;
    }
    case Precision::I8: {
        std::uniform_int_distribution<int8_t> dist(static_cast<int8_t>(max), static_cast<int8_t>(min));
        fill_<int8_t>(blob, rd, dist);
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << blob->getTensorDesc().getPrecision();
    }
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

Blob::Ptr genBlobUniform(const TensorDesc& desc, std::default_random_engine& rd, float min, float max) {
    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    fillUniform(blob, rd, min, max);

    return blob;
}

Blob::Ptr genBlobNormal(const TensorDesc& desc, std::default_random_engine& rd, float mean, float stddev) {
    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    fillNormal(blob, rd, mean, stddev);

    return blob;
}

Blob::Ptr makeScalarBlob(float val, const Precision& precision, size_t numDims) {
    const auto dims = SizeVector(numDims, 1);
    const auto outDesc = TensorDesc(precision, dims, TensorDesc::getLayoutByDims(dims));
    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    switch (precision) {
    case Precision::FP32: {
        const auto outPtr = out->buffer().as<float*>();
        IE_ASSERT(outPtr != nullptr);
        *outPtr = val;
        break;
    }
    case Precision::FP16: {
        const auto outPtr = out->buffer().as<ie_fp16*>();
        IE_ASSERT(outPtr != nullptr);
        *outPtr = PrecisionUtils::f32tof16(val);
        break;
    }
    case Precision::I32: {
        const auto outPtr = out->buffer().as<int32_t*>();
        IE_ASSERT(outPtr != nullptr);
        *outPtr = static_cast<int32_t>(val);
        break;
    }
    case Precision::U8: {
        const auto outPtr = out->buffer().as<uint8_t*>();
        IE_ASSERT(outPtr != nullptr);
        *outPtr = static_cast<uint8_t>(val);
        break;
    }
    case Precision::I8: {
        const auto outPtr = out->buffer().as<int8_t*>();
        IE_ASSERT(outPtr != nullptr);
        *outPtr = static_cast<int8_t>(val);
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << precision;
    }

    return out;
}

Blob::Ptr toFP32(const Blob::Ptr& in) {
    IE_ASSERT(in != nullptr);

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getPrecision() == Precision::FP32) {
        return in;
    }

    const auto outDesc = TensorDesc(Precision::FP32, inDesc.getDims(), inDesc.getLayout());
    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    const auto outPtr = out->buffer().as<float*>();
    IE_ASSERT(outPtr != nullptr);

    switch (inDesc.getPrecision()) {
    case Precision::FP16: {
        const auto inPtr = in->cbuffer().as<const ie_fp16*>();
        IE_ASSERT(inPtr != nullptr);

        PrecisionUtils::f16tof32Arrays(outPtr, inPtr, in->size());
        break;
    }
    case Precision::I32: {
        const auto inPtr = in->cbuffer().as<const int32_t*>();
        IE_ASSERT(inPtr != nullptr);

        std::copy_n(inPtr, in->size(), outPtr);
        break;
    }
    case Precision::U8: {
        const auto inPtr = in->cbuffer().as<const uint8_t*>();
        IE_ASSERT(inPtr != nullptr);

        std::copy_n(inPtr, in->size(), outPtr);
        break;
    }
    case Precision::I8: {
        const auto inPtr = in->cbuffer().as<const int8_t*>();
        IE_ASSERT(inPtr != nullptr);

        std::copy_n(inPtr, in->size(), outPtr);
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << inDesc.getPrecision();
    }

    return out;
}

Blob::Ptr toFP16(const Blob::Ptr& in) {
    IE_ASSERT(in != nullptr);

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getPrecision() == Precision::FP16) {
        return in;
    }

    const auto inFP32 = toFP32(in);

    const auto outDesc = TensorDesc(Precision::FP16, inDesc.getDims(), inDesc.getLayout());
    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    const auto inPtr = inFP32->cbuffer().as<const float*>();
    IE_ASSERT(inPtr != nullptr);

    const auto outPtr = out->buffer().as<ie_fp16*>();
    IE_ASSERT(outPtr != nullptr);

    PrecisionUtils::f32tof16Arrays(outPtr, inPtr, in->size());

    return out;
}

Blob::Ptr toDefLayout(const Blob::Ptr& in) {
    IE_ASSERT(in != nullptr);

    const auto& inDesc = in->getTensorDesc();
    const auto defLayout = TensorDesc::getLayoutByDims(inDesc.getDims());

    if (inDesc.getLayout() == defLayout) {
        return in;
    }

    const auto outDesc = TensorDesc(inDesc.getPrecision(), inDesc.getDims(), defLayout);
    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    blob_copy(in, out);

    return out;
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
