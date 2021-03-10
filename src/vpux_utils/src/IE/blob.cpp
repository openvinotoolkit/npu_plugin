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
// makeBlob
//

MemoryBlob::Ptr vpux::makeBlob(const TensorDesc& desc, const std::shared_ptr<IAllocator>& allocator, void* ptr) {
    MemoryBlob::Ptr out;

    if (ptr != nullptr && allocator == nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(desc, ptr));
    } else if (ptr == nullptr && allocator != nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(desc, allocator));
    } else if (ptr == nullptr && allocator == nullptr) {
        out = as<MemoryBlob>(make_blob_with_precision(desc));
    } else {
        VPUX_THROW("Unsupported case (ptr != NULL && allocator != NULL)");
    }
    out->allocate();

    return out;
}

//
// copyBlob
//

namespace {

bool isCompact(const MemoryBlob::Ptr& blob) {
    const auto& desc = blob->getTensorDesc();
    const auto compactBlkDesc = BlockingDesc(desc.getDims(), desc.getLayout());
    return desc.getBlockingDesc() == compactBlkDesc;
}

}  // namespace

void vpux::copyBlob(const MemoryBlob::Ptr& in, const MemoryBlob::Ptr& out) {
    VPUX_THROW_UNLESS(in != nullptr && out != nullptr, "Got NULL pointer");
    VPUX_THROW_UNLESS(in->getTensorDesc() == out->getTensorDesc(), "Mismatch in TensorDesc");
    VPUX_THROW_UNLESS(isCompact(in) && isCompact(out), "Got non-compact blobs");

    const auto inMem = in->rmap();
    const auto outMem = out->wmap();

    const auto inPtr = inMem.as<const uint8_t*>();
    VPUX_THROW_UNLESS(inPtr != nullptr, "Blob was not allocated");

    const auto outPtr = outMem.as<uint8_t*>();
    VPUX_THROW_UNLESS(outPtr != nullptr, "Blob was not allocated");

    std::copy_n(inPtr, in->byteSize(), outPtr);
}

MemoryBlob::Ptr vpux::copyBlob(const MemoryBlob::Ptr& in, const std::shared_ptr<IAllocator>& allocator) {
    VPUX_THROW_UNLESS(in != nullptr && allocator != nullptr, "Got NULL pointer");
    const auto out = as<MemoryBlob>(make_blob_with_precision(in->getTensorDesc(), allocator));
    out->allocate();
    copyBlob(in, out);
    return out;
}

MemoryBlob::Ptr vpux::copyBlob(const MemoryBlob::Ptr& in, void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr && ptr != nullptr, "Got NULL pointer");
    const auto out = as<MemoryBlob>(make_blob_with_precision(in->getTensorDesc(), ptr));
    out->allocate();
    copyBlob(in, out);
    return out;
}

//
// cvtBlobPrecision
//

namespace {

template <typename InT, typename OutT>
void cvtBlobPrecisionImpl(const MemoryBlob::Ptr& in, const MemoryBlob::Ptr& out) {
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

}  // namespace

void vpux::cvtBlobPrecision(const MemoryBlob::Ptr& in, const MemoryBlob::Ptr& out) {
    VPUX_THROW_UNLESS(in != nullptr && out != nullptr, "Got NULL pointer");
    VPUX_THROW_UNLESS(isCompact(in) && isCompact(out), "Got non-compact blobs");

    const auto& inDesc = in->getTensorDesc();
    const auto& outDesc = out->getTensorDesc();
    VPUX_THROW_UNLESS(inDesc.getDims() == outDesc.getDims(), "Mismatch in Dims");
    VPUX_THROW_UNLESS(inDesc.getLayout() == outDesc.getLayout(), "Mismatch in Layout");

    const auto& inPrecision = inDesc.getPrecision();
    const auto& outPrecision = outDesc.getPrecision();

    if (inPrecision == outPrecision) {
        copyBlob(in, out);
        return;
    }

#define CASE(InT, OutT)                       \
    cvtBlobPrecisionImpl<InT, OutT>(in, out); \
    break

    switch (inPrecision) {
    case Precision::FP64: {
        switch (outPrecision) {
        case Precision::FP32:
            CASE(double, float);
        case Precision::U64:
            CASE(double, uint64_t);
        case Precision::I64:
            CASE(double, int64_t);
        case Precision::U32:
            CASE(double, uint32_t);
        case Precision::I32:
            CASE(double, int32_t);
        case Precision::U16:
            CASE(double, uint16_t);
        case Precision::I16:
            CASE(double, int16_t);
        case Precision::U8:
            CASE(double, uint8_t);
        case Precision::I8:
            CASE(double, int8_t);
        case Precision::FP16:
            CASE(double, float16);
        case Precision::BF16:
            CASE(double, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::FP32: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(float, double);
        case Precision::U64:
            CASE(float, uint64_t);
        case Precision::I64:
            CASE(float, int64_t);
        case Precision::U32:
            CASE(float, uint32_t);
        case Precision::I32:
            CASE(float, int32_t);
        case Precision::U16:
            CASE(float, uint16_t);
        case Precision::I16:
            CASE(float, int16_t);
        case Precision::U8:
            CASE(float, uint8_t);
        case Precision::I8:
            CASE(float, int8_t);
        case Precision::FP16:
            CASE(float, float16);
        case Precision::BF16:
            CASE(float, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::FP16: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(float16, double);
        case Precision::FP32:
            CASE(float16, float);
        case Precision::BF16:
            CASE(float16, bfloat16);
        case Precision::U64:
            CASE(float16, uint64_t);
        case Precision::I64:
            CASE(float16, int64_t);
        case Precision::U32:
            CASE(float16, uint32_t);
        case Precision::I32:
            CASE(float16, int32_t);
        case Precision::U16:
            CASE(float16, uint16_t);
        case Precision::I16:
            CASE(float16, int16_t);
        case Precision::U8:
            CASE(float16, uint8_t);
        case Precision::I8:
            CASE(float16, int8_t);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::BF16: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(bfloat16, double);
        case Precision::FP32:
            CASE(bfloat16, float);
        case Precision::FP16:
            CASE(bfloat16, float16);
        case Precision::U64:
            CASE(bfloat16, uint64_t);
        case Precision::I64:
            CASE(bfloat16, int64_t);
        case Precision::U32:
            CASE(bfloat16, uint32_t);
        case Precision::I32:
            CASE(bfloat16, int32_t);
        case Precision::U16:
            CASE(bfloat16, uint16_t);
        case Precision::I16:
            CASE(bfloat16, int16_t);
        case Precision::U8:
            CASE(bfloat16, uint8_t);
        case Precision::I8:
            CASE(bfloat16, int8_t);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U64: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(uint64_t, double);
        case Precision::FP32:
            CASE(uint64_t, float);
        case Precision::I64:
            CASE(uint64_t, int64_t);
        case Precision::U32:
            CASE(uint64_t, uint32_t);
        case Precision::I32:
            CASE(uint64_t, int32_t);
        case Precision::U16:
            CASE(uint64_t, uint16_t);
        case Precision::I16:
            CASE(uint64_t, int16_t);
        case Precision::U8:
            CASE(uint64_t, uint8_t);
        case Precision::I8:
            CASE(uint64_t, int8_t);
        case Precision::FP16:
            CASE(uint64_t, float16);
        case Precision::BF16:
            CASE(uint64_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I64: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(int64_t, double);
        case Precision::FP32:
            CASE(int64_t, float);
        case Precision::U64:
            CASE(int64_t, uint64_t);
        case Precision::U32:
            CASE(int64_t, uint32_t);
        case Precision::I32:
            CASE(int64_t, int32_t);
        case Precision::U16:
            CASE(int64_t, uint16_t);
        case Precision::I16:
            CASE(int64_t, int16_t);
        case Precision::U8:
            CASE(int64_t, uint8_t);
        case Precision::I8:
            CASE(int64_t, int8_t);
        case Precision::FP16:
            CASE(int64_t, float16);
        case Precision::BF16:
            CASE(int64_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U32: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(uint32_t, double);
        case Precision::FP32:
            CASE(uint32_t, float);
        case Precision::U64:
            CASE(uint32_t, uint64_t);
        case Precision::I64:
            CASE(uint32_t, int64_t);
        case Precision::I32:
            CASE(uint32_t, int32_t);
        case Precision::U16:
            CASE(uint32_t, uint16_t);
        case Precision::I16:
            CASE(uint32_t, int16_t);
        case Precision::U8:
            CASE(uint32_t, uint8_t);
        case Precision::I8:
            CASE(uint32_t, int8_t);
        case Precision::FP16:
            CASE(uint32_t, float16);
        case Precision::BF16:
            CASE(uint32_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I32: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(int32_t, double);
        case Precision::FP32:
            CASE(int32_t, float);
        case Precision::U64:
            CASE(int32_t, uint64_t);
        case Precision::I64:
            CASE(int32_t, int64_t);
        case Precision::U32:
            CASE(int32_t, uint32_t);
        case Precision::U16:
            CASE(int32_t, uint16_t);
        case Precision::I16:
            CASE(int32_t, int16_t);
        case Precision::U8:
            CASE(int32_t, uint8_t);
        case Precision::I8:
            CASE(int32_t, int8_t);
        case Precision::FP16:
            CASE(int32_t, float16);
        case Precision::BF16:
            CASE(int32_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U16: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(uint16_t, double);
        case Precision::FP32:
            CASE(uint16_t, float);
        case Precision::U64:
            CASE(uint16_t, uint64_t);
        case Precision::I64:
            CASE(uint16_t, int64_t);
        case Precision::U32:
            CASE(uint16_t, uint32_t);
        case Precision::I32:
            CASE(uint16_t, int32_t);
        case Precision::I16:
            CASE(uint16_t, int16_t);
        case Precision::U8:
            CASE(uint16_t, uint8_t);
        case Precision::I8:
            CASE(uint16_t, int8_t);
        case Precision::FP16:
            CASE(uint16_t, float16);
        case Precision::BF16:
            CASE(uint16_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I16: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(int16_t, double);
        case Precision::FP32:
            CASE(int16_t, float);
        case Precision::U64:
            CASE(int16_t, uint64_t);
        case Precision::I64:
            CASE(int16_t, int64_t);
        case Precision::U32:
            CASE(int16_t, uint32_t);
        case Precision::I32:
            CASE(int16_t, int32_t);
        case Precision::U16:
            CASE(int16_t, uint16_t);
        case Precision::U8:
            CASE(int16_t, uint8_t);
        case Precision::I8:
            CASE(int16_t, int8_t);
        case Precision::FP16:
            CASE(int16_t, float16);
        case Precision::BF16:
            CASE(int16_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::U8: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(uint8_t, double);
        case Precision::FP32:
            CASE(uint8_t, float);
        case Precision::U64:
            CASE(uint8_t, uint64_t);
        case Precision::I64:
            CASE(uint8_t, int64_t);
        case Precision::U32:
            CASE(uint8_t, uint32_t);
        case Precision::I32:
            CASE(uint8_t, int32_t);
        case Precision::U16:
            CASE(uint8_t, uint16_t);
        case Precision::I16:
            CASE(uint8_t, int16_t);
        case Precision::I8:
            CASE(uint8_t, int8_t);
        case Precision::FP16:
            CASE(uint8_t, float16);
        case Precision::BF16:
            CASE(uint8_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    case Precision::I8: {
        switch (outPrecision) {
        case Precision::FP64:
            CASE(int8_t, double);
        case Precision::FP32:
            CASE(int8_t, float);
        case Precision::U64:
            CASE(int8_t, uint64_t);
        case Precision::I64:
            CASE(int8_t, int64_t);
        case Precision::U32:
            CASE(int8_t, uint32_t);
        case Precision::I32:
            CASE(int8_t, int32_t);
        case Precision::U16:
            CASE(int8_t, uint16_t);
        case Precision::I16:
            CASE(int8_t, int16_t);
        case Precision::U8:
            CASE(int8_t, uint8_t);
        case Precision::FP16:
            CASE(int8_t, float16);
        case Precision::BF16:
            CASE(int8_t, bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
        }
        break;
    }
    default:
        VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision, outPrecision);
    }

#undef CASE
}

MemoryBlob::Ptr vpux::toPrecision(const MemoryBlob::Ptr& in, const Precision& precision,
                                  const std::shared_ptr<IAllocator>& allocator, void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getPrecision() == precision && allocator == nullptr && ptr == nullptr) {
        return in;
    }

    const auto outDesc = TensorDesc(precision, inDesc.getDims(), inDesc.getLayout());
    const auto out = makeBlob(outDesc, allocator, ptr);

    cvtBlobPrecision(in, out);

    return out;
}

MemoryBlob::Ptr vpux::toDefPrecision(const MemoryBlob::Ptr& in, const std::shared_ptr<IAllocator>& allocator,
                                     void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");

    const auto inPrec = in->getTensorDesc().getPrecision();

    if (inPrec == Precision::U8 || inPrec == Precision::FP16) {
        return toPrecision(in, Precision::FP32, allocator, ptr);
    } else {
        if (allocator == nullptr && ptr == nullptr) {
            return in;
        } else {
            return allocator != nullptr ? copyBlob(in, allocator) : copyBlob(in, ptr);
        }
    }
}

//
// cvtBlobLayout
//

void vpux::cvtBlobLayout(const MemoryBlob::Ptr& in, const MemoryBlob::Ptr& out) {
    VPUX_THROW_UNLESS(in != nullptr && out != nullptr, "Got NULL pointer");

    const auto& inDesc = in->getTensorDesc();
    const auto& outDesc = out->getTensorDesc();
    VPUX_THROW_UNLESS(inDesc.getDims() == outDesc.getDims(), "Mismatch in Dims");
    VPUX_THROW_UNLESS(inDesc.getPrecision() == outDesc.getPrecision(), "Mismatch in Precision");

    const auto& inLayout = inDesc.getLayout();
    const auto& outLayout = outDesc.getLayout();

    if (inLayout == outLayout) {
        copyBlob(in, out);
        return;
    }

    blob_copy(in, out);
}

MemoryBlob::Ptr vpux::toLayout(const MemoryBlob::Ptr& in, Layout layout, const std::shared_ptr<IAllocator>& allocator,
                               void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getLayout() == layout && allocator == nullptr && ptr == nullptr) {
        return in;
    }

    const auto outDesc = TensorDesc(inDesc.getPrecision(), inDesc.getDims(), layout);
    const auto out = makeBlob(outDesc, allocator, ptr);

    cvtBlobLayout(in, out);

    return out;
}

MemoryBlob::Ptr vpux::toDefLayout(const MemoryBlob::Ptr& in, const std::shared_ptr<IAllocator>& allocator, void* ptr) {
    VPUX_THROW_UNLESS(in != nullptr, "Got NULL pointer");
    const auto& inDesc = in->getTensorDesc();
    const auto defLayout = TensorDesc::getLayoutByDims(inDesc.getDims());
    return toLayout(in, defLayout, allocator, ptr);
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
