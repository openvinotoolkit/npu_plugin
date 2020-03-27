//
// Copyright 2020 Intel Corporation.
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

#include <ie_utils.hpp>

#include <precision_utils.h>
#include <blob_factory.hpp>
#include <blob_transform.hpp>

//
// makeSingleValueBlob
//

namespace {

template <typename T>
ie::Blob::Ptr makeSingleValueBlob_(const ie::TensorDesc& desc, T val) {
    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    switch (desc.getPrecision()) {
    case ie::Precision::FP32: {
        const auto outPtr = blob->buffer().as<float*>();
        IE_ASSERT(outPtr != nullptr);
        std::fill_n(outPtr, blob->size(), static_cast<float>(val));
        break;
    }
    case ie::Precision::FP16: {
        const auto outPtr = blob->buffer().as<ie::ie_fp16*>();
        IE_ASSERT(outPtr != nullptr);
        std::fill_n(outPtr, blob->size(), ie::PrecisionUtils::f32tof16(static_cast<float>(val)));
        break;
    }
    case ie::Precision::I64: {
        const auto outPtr = blob->buffer().as<int64_t*>();
        IE_ASSERT(outPtr != nullptr);
        std::fill_n(outPtr, blob->size(), static_cast<int64_t>(val));
        break;
    }
    case ie::Precision::I32: {
        const auto outPtr = blob->buffer().as<int32_t*>();
        IE_ASSERT(outPtr != nullptr);
        std::fill_n(outPtr, blob->size(), static_cast<int32_t>(val));
        break;
    }
    case ie::Precision::U16: {
        const auto outPtr = blob->buffer().as<uint16_t*>();
        IE_ASSERT(outPtr != nullptr);
        std::fill_n(outPtr, blob->size(), static_cast<uint16_t>(val));
        break;
    }
    case ie::Precision::U8: {
        const auto outPtr = blob->buffer().as<uint8_t*>();
        IE_ASSERT(outPtr != nullptr);
        std::fill_n(outPtr, blob->size(), static_cast<uint8_t>(val));
        break;
    }
    case ie::Precision::I8: {
        const auto outPtr = blob->buffer().as<int8_t*>();
        IE_ASSERT(outPtr != nullptr);
        std::fill_n(outPtr, blob->size(), static_cast<int8_t>(val));
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << desc.getPrecision();
    }

    return blob;
}

}  // namespace

ie::Blob::Ptr makeSingleValueBlob(const ie::TensorDesc& desc, float val) {
    return makeSingleValueBlob_(desc, val);
}

ie::Blob::Ptr makeSingleValueBlob(const ie::TensorDesc& desc, int64_t val) {
    return makeSingleValueBlob_(desc, val);
}

//
// makeScalarBlob
//

namespace {

template <typename T>
ie::Blob::Ptr makeScalarBlob_(T val, const ie::Precision& precision, size_t numDims) {
    const auto dims = ie::SizeVector(numDims, 1);
    const auto desc = ie::TensorDesc(precision, dims, ie::TensorDesc::getLayoutByDims(dims));
    return makeSingleValueBlob(desc, val);
}

}  // namespace

ie::Blob::Ptr makeScalarBlob(float val, const ie::Precision& precision, size_t numDims) {
    return makeScalarBlob_(val, precision, numDims);
}

ie::Blob::Ptr makeScalarBlob(int64_t val, const ie::Precision& precision, size_t numDims) {
    return makeScalarBlob_(val, precision, numDims);
}

//
// toPrecision
//

namespace {

template <typename InT, typename OutT>
void cvtBlobPrecision_(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out) {
    IE_ASSERT(in->getTensorDesc().getPrecision().size() == sizeof(InT));
    IE_ASSERT(out->getTensorDesc().getPrecision().size() == sizeof(OutT));

    const auto inPtr = in->cbuffer().as<const InT*>();
    IE_ASSERT(inPtr != nullptr);

    const auto outPtr = out->buffer().as<OutT*>();
    IE_ASSERT(outPtr != nullptr);

    std::copy_n(inPtr, in->size(), outPtr);
}

template <>
void cvtBlobPrecision_<ie::ie_fp16, float>(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out) {
    IE_ASSERT(in->getTensorDesc().getPrecision().size() == sizeof(ie::ie_fp16));
    IE_ASSERT(out->getTensorDesc().getPrecision().size() == sizeof(float));

    const auto inPtr = in->cbuffer().as<const ie::ie_fp16*>();
    IE_ASSERT(inPtr != nullptr);

    const auto outPtr = out->buffer().as<float*>();
    IE_ASSERT(outPtr != nullptr);

    ie::PrecisionUtils::f16tof32Arrays(outPtr, inPtr, in->size());
}

template <>
void cvtBlobPrecision_<float, ie::ie_fp16>(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out) {
    IE_ASSERT(in->getTensorDesc().getPrecision().size() == sizeof(float));
    IE_ASSERT(out->getTensorDesc().getPrecision().size() == sizeof(ie::ie_fp16));

    const auto inPtr = in->cbuffer().as<const float*>();
    IE_ASSERT(inPtr != nullptr);

    const auto outPtr = out->buffer().as<ie::ie_fp16*>();
    IE_ASSERT(outPtr != nullptr);

    ie::PrecisionUtils::f32tof16Arrays(outPtr, inPtr, in->size());
}

}  // namespace

ie::Blob::Ptr toPrecision(const ie::Blob::Ptr& in, const ie::Precision& precision) {
    IE_ASSERT(in != nullptr);

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getPrecision() == precision) {
        return in;
    }

    const auto outDesc = ie::TensorDesc(precision, inDesc.getDims(), inDesc.getLayout());
    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    IE_ASSERT(in->getTensorDesc().getDims() == out->getTensorDesc().getDims());
    IE_ASSERT(in->getTensorDesc().getLayout() == out->getTensorDesc().getLayout());

    const auto& inPrecision = in->getTensorDesc().getPrecision();
    const auto& outPrecision = out->getTensorDesc().getPrecision();

    switch (inPrecision) {
    case ie::Precision::FP32: {
        switch (outPrecision) {
        case ie::Precision::FP16: {
            cvtBlobPrecision_<float, ie::ie_fp16>(in, out);
            break;
        }
        case ie::Precision::I64: {
            cvtBlobPrecision_<float, int64_t>(in, out);
            break;
        }
        case ie::Precision::I32: {
            cvtBlobPrecision_<float, int32_t>(in, out);
            break;
        }
        case ie::Precision::U16: {
            cvtBlobPrecision_<float, uint16_t>(in, out);
            break;
        }
        case ie::Precision::U8: {
            cvtBlobPrecision_<float, uint8_t>(in, out);
            break;
        }
        case ie::Precision::I8: {
            cvtBlobPrecision_<float, int8_t>(in, out);
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported combination of precisions " << inPrecision << " -> " << outPrecision;
        }
        break;
    }
    case ie::Precision::FP16: {
        switch (outPrecision) {
        case ie::Precision::FP32: {
            cvtBlobPrecision_<ie::ie_fp16, float>(in, out);
            break;
        }
        case ie::Precision::I64: {
            cvtBlobPrecision_<float, int64_t>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::I32: {
            cvtBlobPrecision_<float, int32_t>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::U16: {
            cvtBlobPrecision_<float, uint16_t>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::U8: {
            cvtBlobPrecision_<float, uint8_t>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::I8: {
            cvtBlobPrecision_<float, int8_t>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported combination of precisions " << inPrecision << " -> " << outPrecision;
        }
        break;
    }
    case ie::Precision::I64: {
        switch (outPrecision) {
        case ie::Precision::FP32: {
            cvtBlobPrecision_<int64_t, float>(in, out);
            break;
        }
        case ie::Precision::FP16: {
            cvtBlobPrecision_<float, ie::ie_fp16>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::I32: {
            cvtBlobPrecision_<int64_t, int32_t>(in, out);
            break;
        }
        case ie::Precision::U8: {
            cvtBlobPrecision_<int64_t, uint8_t>(in, out);
            break;
        }
        case ie::Precision::I8: {
            cvtBlobPrecision_<int64_t, int8_t>(in, out);
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported combination of precisions " << inPrecision << " -> " << outPrecision;
        }
        break;
    }
    case ie::Precision::I32: {
        switch (outPrecision) {
        case ie::Precision::FP32: {
            cvtBlobPrecision_<int32_t, float>(in, out);
            break;
        }
        case ie::Precision::FP16: {
            cvtBlobPrecision_<float, ie::ie_fp16>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::I64: {
            cvtBlobPrecision_<int32_t, int64_t>(in, out);
            break;
        }
        case ie::Precision::U16: {
            cvtBlobPrecision_<int32_t, uint16_t>(in, out);
            break;
        }
        case ie::Precision::U8: {
            cvtBlobPrecision_<int32_t, uint8_t>(in, out);
            break;
        }
        case ie::Precision::I8: {
            cvtBlobPrecision_<int32_t, int8_t>(in, out);
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported combination of precisions " << inPrecision << " -> " << outPrecision;
        }
        break;
    }
    case ie::Precision::U8: {
        switch (outPrecision) {
        case ie::Precision::FP32: {
            cvtBlobPrecision_<uint8_t, float>(in, out);
            break;
        }
        case ie::Precision::FP16: {
            cvtBlobPrecision_<float, ie::ie_fp16>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::I64: {
            cvtBlobPrecision_<uint8_t, int64_t>(in, out);
            break;
        }
        case ie::Precision::I32: {
            cvtBlobPrecision_<uint8_t, int32_t>(in, out);
            break;
        }
        case ie::Precision::U16: {
            cvtBlobPrecision_<uint8_t, uint16_t>(in, out);
            break;
        }
        case ie::Precision::I8: {
            cvtBlobPrecision_<uint8_t, int8_t>(in, out);
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported combination of precisions " << inPrecision << " -> " << outPrecision;
        }
        break;
    }
    case ie::Precision::I8: {
        switch (outPrecision) {
        case ie::Precision::FP32: {
            cvtBlobPrecision_<int8_t, float>(in, out);
            break;
        }
        case ie::Precision::FP16: {
            cvtBlobPrecision_<float, ie::ie_fp16>(toPrecision(in, ie::Precision::FP32), out);
            break;
        }
        case ie::Precision::I64: {
            cvtBlobPrecision_<int8_t, int64_t>(in, out);
            break;
        }
        case ie::Precision::I32: {
            cvtBlobPrecision_<int8_t, int32_t>(in, out);
            break;
        }
        case ie::Precision::U16: {
            cvtBlobPrecision_<int8_t, uint16_t>(in, out);
            break;
        }
        case ie::Precision::U8: {
            cvtBlobPrecision_<int8_t, uint8_t>(in, out);
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported combination of precisions " << inPrecision << " -> " << outPrecision;
        }
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Unsupported combination of precisions " << inPrecision << " -> " << outPrecision;
    }

    return out;
}

ie::Blob::Ptr toDefPrecision(const ie::Blob::Ptr& in) {
    IE_ASSERT(in != nullptr);

    const auto inPrec = in->getTensorDesc().getPrecision();

    if (inPrec == ie::Precision::U8 || inPrec == ie::Precision::FP16) {
        return toPrecision(in, ie::Precision::FP32);
    } else {
        return in;
    }
}

//
// toLayout
//

ie::Blob::Ptr toLayout(const ie::Blob::Ptr& in, ie::Layout layout) {
    IE_ASSERT(in != nullptr);

    const auto& inDesc = in->getTensorDesc();
    if (inDesc.getLayout() == layout) {
        return in;
    }

    const auto outDesc = ie::TensorDesc(inDesc.getPrecision(), inDesc.getDims(), layout);
    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    blob_copy(in, out);

    return out;
}

ie::Blob::Ptr toDefLayout(const ie::Blob::Ptr& in) {
    IE_ASSERT(in != nullptr);

    const auto& inDesc = in->getTensorDesc();
    const auto defLayout = ie::TensorDesc::getLayoutByDims(inDesc.getDims());

    return toLayout(in, defLayout);
}
