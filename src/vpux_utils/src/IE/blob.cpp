//
// Copyright Intel Corporation.
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

#include "vpux/utils/IE/blob.hpp"

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <blob_factory.hpp>
#include <blob_transform.hpp>

#include <fstream>

using namespace vpux;
using namespace InferenceEngine;

//
// makeSplatBlob
//

namespace {

template <typename T>
void fillN(T* ptr, size_t size, T val) {
    VPUX_THROW_UNLESS(ptr != nullptr, "NULL pointer");

    loop_1d(LoopExecPolicy::Parallel, size, [ptr, val](size_t i) {
        ptr[i] = val;
    });
}

}  // namespace

MemoryBlob::Ptr vpux::makeSplatBlob(const TensorDesc& desc, double val) {
    const auto blob = as<MemoryBlob>(make_blob_with_precision(desc));
    blob->allocate();

    const auto blobMem = blob->wmap();

    switch (desc.getPrecision()) {
    case Precision::FP64: {
        const auto outPtr = blobMem.as<double*>();
        fillN(outPtr, blob->size(), checked_cast<double>(val));
        break;
    }
    case Precision::FP32: {
        const auto outPtr = blobMem.as<float*>();
        fillN(outPtr, blob->size(), checked_cast<float>(val));
        break;
    }
    case Precision::FP16: {
        const auto outPtr = blobMem.as<float16*>();
        fillN(outPtr, blob->size(), checked_cast<float16>(val));
        break;
    }
    case Precision::BF16: {
        const auto outPtr = blobMem.as<bfloat16*>();
        fillN(outPtr, blob->size(), checked_cast<bfloat16>(val));
        break;
    }
    case Precision::U64: {
        const auto outPtr = blobMem.as<uint64_t*>();
        fillN(outPtr, blob->size(), checked_cast<uint64_t>(val));
        break;
    }
    case Precision::I64: {
        const auto outPtr = blobMem.as<int64_t*>();
        fillN(outPtr, blob->size(), checked_cast<int64_t>(val));
        break;
    }
    case Precision::U32: {
        const auto outPtr = blobMem.as<uint32_t*>();
        fillN(outPtr, blob->size(), checked_cast<uint32_t>(val));
        break;
    }
    case Precision::I32: {
        const auto outPtr = blobMem.as<int32_t*>();
        fillN(outPtr, blob->size(), checked_cast<int32_t>(val));
        break;
    }
    case Precision::U16: {
        const auto outPtr = blobMem.as<uint16_t*>();
        fillN(outPtr, blob->size(), checked_cast<uint16_t>(val));
        break;
    }
    case Precision::I16: {
        const auto outPtr = blobMem.as<int16_t*>();
        fillN(outPtr, blob->size(), checked_cast<int16_t>(val));
        break;
    }
    case Precision::U8: {
        const auto outPtr = blobMem.as<uint8_t*>();
        fillN(outPtr, blob->size(), checked_cast<uint8_t>(val));
        break;
    }
    case Precision::I8: {
        const auto outPtr = blobMem.as<int8_t*>();
        fillN(outPtr, blob->size(), checked_cast<int8_t>(val));
        break;
    }
    default:
        VPUX_THROW("Unsupported precision : {0}", desc.getPrecision());
    }

    return blob;
}

//
// makeScalarBlob
//

MemoryBlob::Ptr vpux::makeScalarBlob(double val, const Precision& precision, size_t numDims) {
    const auto dims = SizeVector(numDims, 1);
    const auto desc = TensorDesc(precision, dims, TensorDesc::getLayoutByDims(dims));
    return makeSplatBlob(desc, val);
}

//
// toPrecision
//

namespace {

template <typename InT, typename OutT>
void cvtBlobPrecision(const MemoryBlob::Ptr& in, const MemoryBlob::Ptr& out) {
    const auto& inPresision = in->getTensorDesc().getPrecision();
    const auto& outPresision = out->getTensorDesc().getPrecision();

    VPUX_THROW_UNLESS(inPresision.size() == sizeof(InT), "Wrong blob precision : {0}", inPresision);
    VPUX_THROW_UNLESS(outPresision.size() == sizeof(OutT), "Wrong blob precision : {0}", outPresision);

    const auto inMem = in->rmap();
    const auto outMem = out->wmap();

    const auto inPtr = inMem.as<const InT*>();
    VPUX_THROW_UNLESS(inPtr != nullptr, "Blob was not allocated");

    const auto outPtr = outMem.as<OutT*>();
    VPUX_THROW_UNLESS(outPtr != nullptr, "Blob was not allocated");

    loop_1d(LoopExecPolicy::Parallel, in->size(), [inPtr, outPtr](int64_t i) {
        outPtr[i] = checked_cast<OutT>(inPtr[i]);
    });
}

template <Precision::ePrecision InP, Precision::ePrecision OutP>
void cvtBlobPrecisionImpl(const MemoryBlob::Ptr& in, const MemoryBlob::Ptr& out) {
    using InT = typename PrecisionTrait<InP>::value_type;
    using OutT = typename PrecisionTrait<OutP>::value_type;
    cvtBlobPrecision<InT, OutT>(in, out);
}

}  // namespace

MemoryBlob::Ptr vpux::toPrecision(const MemoryBlob::Ptr& in, const Precision& precision,
                                  const std::shared_ptr<InferenceEngine::IAllocator>& allocator, void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getPrecision() == precision) {
        return in;
    }

    const auto outDesc = TensorDesc(precision, inDesc.getDims(), inDesc.getLayout());

    MemoryBlob::Ptr out;
    if (ptr != nullptr && allocator == nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(outDesc, ptr));
    } else if (ptr == nullptr && allocator != nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(outDesc, allocator));
    } else if (ptr == nullptr && allocator == nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(outDesc));
    } else {
        VPUX_THROW("Unsupported case (ptr != NULL && allocator != NULL)");
    }
    out->allocate();

    const auto& inPrecision = in->getTensorDesc().getPrecision();
    const auto& outPrecision = out->getTensorDesc().getPrecision();

#define CASE(InP, OutP)                                             \
    cvtBlobPrecisionImpl<Precision::InP, Precision::OutP>(in, out); \
    break

    switch (inPrecision) {
    case Precision::FP64: {
        switch (outPrecision) {
        case Precision::FP32:
            CASE(FP64, FP32);
        case Precision::U64:
            CASE(FP64, U64);
        case Precision::I64:
            CASE(FP64, I64);
        case Precision::U32:
            CASE(FP64, U32);
        case Precision::I32:
            CASE(FP64, I32);
        case Precision::U16:
            CASE(FP64, U16);
        case Precision::I16:
            CASE(FP64, I16);
        case Precision::U8:
            CASE(FP64, U8);
        case Precision::I8:
            CASE(FP64, I8);
        case Precision::FP16:
            cvtBlobPrecision<double, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<double, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::FP32: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(FP32, FP64);
        case Precision::U64:
            CASE(FP32, U64);
        case Precision::I64:
            CASE(FP32, I64);
        case Precision::U32:
            CASE(FP32, U32);
        case Precision::I32:
            CASE(FP32, I32);
        case Precision::U16:
            CASE(FP32, U16);
        case Precision::I16:
            CASE(FP32, I16);
        case Precision::U8:
            CASE(FP32, U8);
        case Precision::I8:
            CASE(FP32, I8);
        case Precision::FP16:
            cvtBlobPrecision<float, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<float, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::FP16: {
        switch (outPrecision) {
        case Precision::FP64:
            cvtBlobPrecision<float16, double>(in, out);
            break;
        case Precision::FP32:
            cvtBlobPrecision<float16, float>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<float16, bfloat16>(in, out);
            break;
        case Precision::U64:
            cvtBlobPrecision<float16, uint64_t>(in, out);
            break;
        case Precision::I64:
            cvtBlobPrecision<float16, int64_t>(in, out);
            break;
        case Precision::U32:
            cvtBlobPrecision<float16, uint32_t>(in, out);
            break;
        case Precision::I32:
            cvtBlobPrecision<float16, int32_t>(in, out);
            break;
        case Precision::U16:
            cvtBlobPrecision<float16, uint16_t>(in, out);
            break;
        case Precision::I16:
            cvtBlobPrecision<float16, int16_t>(in, out);
            break;
        case Precision::U8:
            cvtBlobPrecision<float16, uint8_t>(in, out);
            break;
        case Precision::I8:
            cvtBlobPrecision<float16, int8_t>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::BF16: {
        switch (outPrecision) {
        case Precision::FP64:
            cvtBlobPrecision<bfloat16, double>(in, out);
            break;
        case Precision::FP32:
            cvtBlobPrecision<bfloat16, float>(in, out);
            break;
        case Precision::FP16:
            cvtBlobPrecision<bfloat16, float16>(in, out);
            break;
        case Precision::U64:
            cvtBlobPrecision<bfloat16, uint64_t>(in, out);
            break;
        case Precision::I64:
            cvtBlobPrecision<bfloat16, int64_t>(in, out);
            break;
        case Precision::U32:
            cvtBlobPrecision<bfloat16, uint32_t>(in, out);
            break;
        case Precision::I32:
            cvtBlobPrecision<bfloat16, int32_t>(in, out);
            break;
        case Precision::U16:
            cvtBlobPrecision<bfloat16, uint16_t>(in, out);
            break;
        case Precision::I16:
            cvtBlobPrecision<bfloat16, int16_t>(in, out);
            break;
        case Precision::U8:
            cvtBlobPrecision<bfloat16, uint8_t>(in, out);
            break;
        case Precision::I8:
            cvtBlobPrecision<bfloat16, int8_t>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U64: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(U64, FP64);
        case Precision::FP32:
            CASE(U64, FP32);
        case Precision::I64:
            CASE(U64, I64);
        case Precision::U32:
            CASE(U64, U32);
        case Precision::I32:
            CASE(U64, I32);
        case Precision::U16:
            CASE(U64, U16);
        case Precision::I16:
            CASE(U64, I16);
        case Precision::U8:
            CASE(U64, U8);
        case Precision::I8:
            CASE(U64, I8);
        case Precision::FP16:
            cvtBlobPrecision<uint64_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<uint64_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I64: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(I64, FP64);
        case Precision::FP32:
            CASE(I64, FP32);
        case Precision::U64:
            CASE(I64, U64);
        case Precision::U32:
            CASE(I64, U32);
        case Precision::I32:
            CASE(I64, I32);
        case Precision::U16:
            CASE(I64, U16);
        case Precision::I16:
            CASE(I64, I16);
        case Precision::U8:
            CASE(I64, U8);
        case Precision::I8:
            CASE(I64, I8);
        case Precision::FP16:
            cvtBlobPrecision<int64_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<int64_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U32: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(U32, FP64);
        case Precision::FP32:
            CASE(U32, FP32);
        case Precision::U64:
            CASE(U32, U64);
        case Precision::I64:
            CASE(U32, I64);
        case Precision::I32:
            CASE(U32, I32);
        case Precision::U16:
            CASE(U32, U16);
        case Precision::I16:
            CASE(U32, I16);
        case Precision::U8:
            CASE(U32, U8);
        case Precision::I8:
            CASE(U32, I8);
        case Precision::FP16:
            cvtBlobPrecision<uint32_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<uint32_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I32: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(I32, FP64);
        case Precision::FP32:
            CASE(I32, FP32);
        case Precision::U64:
            CASE(I32, U64);
        case Precision::I64:
            CASE(I32, I64);
        case Precision::U32:
            CASE(I32, U32);
        case Precision::U16:
            CASE(I32, U16);
        case Precision::I16:
            CASE(I32, I16);
        case Precision::U8:
            CASE(I32, U8);
        case Precision::I8:
            CASE(I32, I8);
        case Precision::FP16:
            cvtBlobPrecision<int32_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<int32_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U16: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(U16, FP64);
        case Precision::FP32:
            CASE(U16, FP32);
        case Precision::U64:
            CASE(U16, U64);
        case Precision::I64:
            CASE(U16, I64);
        case Precision::U32:
            CASE(U16, U32);
        case Precision::I32:
            CASE(U16, I32);
        case Precision::I16:
            CASE(U16, I16);
        case Precision::U8:
            CASE(U16, U8);
        case Precision::I8:
            CASE(U16, I8);
        case Precision::FP16:
            cvtBlobPrecision<uint16_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<uint16_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I16: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(I16, FP64);
        case Precision::FP32:
            CASE(I16, FP32);
        case Precision::U64:
            CASE(I16, U64);
        case Precision::I64:
            CASE(I16, I64);
        case Precision::U32:
            CASE(I16, U32);
        case Precision::I32:
            CASE(I16, I32);
        case Precision::U16:
            CASE(I16, U16);
        case Precision::U8:
            CASE(I16, U8);
        case Precision::I8:
            CASE(I16, I8);
        case Precision::FP16:
            cvtBlobPrecision<int16_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<int16_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U8: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(U8, FP64);
        case Precision::FP32:
            CASE(U8, FP32);
        case Precision::U64:
            CASE(U8, U64);
        case Precision::I64:
            CASE(U8, I64);
        case Precision::U32:
            CASE(U8, U32);
        case Precision::I32:
            CASE(U8, I32);
        case Precision::U16:
            CASE(U8, U16);
        case Precision::I16:
            CASE(U8, I16);
        case Precision::I8:
            CASE(U8, I8);
        case Precision::FP16:
            cvtBlobPrecision<uint8_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<uint8_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I8: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(I8, FP64);
        case Precision::FP32:
            CASE(I8, FP32);
        case Precision::U64:
            CASE(I8, U64);
        case Precision::I64:
            CASE(I8, I64);
        case Precision::U32:
            CASE(I8, U32);
        case Precision::I32:
            CASE(I8, I32);
        case Precision::U16:
            CASE(I8, U16);
        case Precision::I16:
            CASE(I8, I16);
        case Precision::U8:
            CASE(I8, U8);
        case Precision::FP16:
            cvtBlobPrecision<int8_t, float16>(in, out);
            break;
        case Precision::BF16:
            cvtBlobPrecision<int8_t, bfloat16>(in, out);
            break;
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    default:
        VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
    }

#undef CASE

    return out;
}

MemoryBlob::Ptr vpux::toDefPrecision(const MemoryBlob::Ptr& in,
                                     const std::shared_ptr<InferenceEngine::IAllocator>& allocator, void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");

    const auto inPrec = in->getTensorDesc().getPrecision();

    if (inPrec == Precision::U8 || inPrec == Precision::FP16) {
        return toPrecision(in, Precision::FP32, allocator, ptr);
    } else {
        return in;
    }
}

//
// toLayout
//

MemoryBlob::Ptr vpux::toLayout(const MemoryBlob::Ptr& in, Layout layout,
                               const std::shared_ptr<InferenceEngine::IAllocator>& allocator, void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getLayout() == layout) {
        return in;
    }

    const auto outDesc = TensorDesc(inDesc.getPrecision(), inDesc.getDims(), layout);

    MemoryBlob::Ptr out;
    if (ptr != nullptr && allocator == nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(outDesc, ptr));
    } else if (ptr == nullptr && allocator != nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(outDesc, allocator));
    } else if (ptr == nullptr && allocator == nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(outDesc));
    } else {
        VPUX_THROW("Unsupported case (ptr != NULL && allocator != NULL)");
    }
    out->allocate();

    blob_copy(in, out);

    return out;
}

MemoryBlob::Ptr vpux::toDefLayout(const MemoryBlob::Ptr& in,
                                  const std::shared_ptr<InferenceEngine::IAllocator>& allocator, void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");
    const auto& inDesc = in->getTensorDesc();
    const auto defLayout = TensorDesc::getLayoutByDims(inDesc.getDims());
    return toLayout(in, defLayout, allocator, ptr);
}

//
// reallocateBlob
//

MemoryBlob::Ptr vpux::reallocateBlob(const MemoryBlob::Ptr& in, const std::shared_ptr<IAllocator>& allocator) {
    const auto out = as<MemoryBlob>(make_blob_with_precision(in->getTensorDesc(), allocator));
    out->allocate();

    const auto inMem = in->rmap();
    const auto outMem = out->wmap();

    const auto inPtr = inMem.as<const uint8_t*>();
    VPUX_THROW_UNLESS(inPtr != nullptr, "Blob was not allocated");

    const auto outPtr = outMem.as<uint8_t*>();
    VPUX_THROW_UNLESS(outPtr != nullptr, "Blob was not allocated");

    std::copy_n(inPtr, in->byteSize(), outPtr);

    return out;
}

//
// dumpBlobs
//

void vpux::dumpBlobs(const BlobMap& blobMap, StringRef dstPath, StringRef blobType) {
    auto log = Logger::global();

    if (dstPath.empty()) {
        log.warning("Destination path is empty in dumpBlobs");
        return;
    }

    for (const auto& p : blobMap | map_values | indexed) {
        const auto ind = p.index();
        const auto& blob = as<MemoryBlob>(p.value());
        VPUX_THROW_UNLESS(blob != nullptr, "Got non MemoryBlob");

        const auto filePath = llvm::formatv("{0}/{1}-dump{2}.bin", dstPath, blobType, ind).str();
        std::ofstream file(filePath, std::ios_base::binary);

        if (!file.is_open()) {
            log.warning("Failed to open file {0} for write", filePath);
            continue;
        }

        const auto mem = blob->rmap();
        const auto ptr = mem.as<const char*>();
        VPUX_THROW_UNLESS(ptr != nullptr, "Blob was not allocated");

        file.write(ptr, blob->byteSize());
    }
}

//
// getMemorySize
//

Byte vpux::getMemorySize(const TensorDesc& desc) {
    const Byte elemSize(desc.getPrecision().size());

    const auto& dims = desc.getDims();
    const auto totalNumElements = std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());

    return totalNumElements * elemSize;
}
