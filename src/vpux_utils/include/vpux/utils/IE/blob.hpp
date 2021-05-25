//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <ie_blob.h>
#include <ie_compound_blob.h>
#include <ie_remote_context.hpp>

#include <memory>

namespace vpux {

//
// makeSplatBlob
//

InferenceEngine::MemoryBlob::Ptr makeSplatBlob(const InferenceEngine::TensorDesc& desc, double val);

//
// makeScalarBlob
//

InferenceEngine::MemoryBlob::Ptr makeScalarBlob(
        double val, const InferenceEngine::Precision& precision = InferenceEngine::Precision::FP32, size_t numDims = 1);

//
// makeBlob
//

InferenceEngine::MemoryBlob::Ptr makeBlob(const InferenceEngine::TensorDesc& desc,
                                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr,
                                          void* ptr = nullptr);

//
// copyBlob
//

void copyBlob(const InferenceEngine::MemoryBlob::Ptr& in, const InferenceEngine::MemoryBlob::Ptr& out);

InferenceEngine::MemoryBlob::Ptr copyBlob(const InferenceEngine::MemoryBlob::Ptr& in,
                                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator);

InferenceEngine::MemoryBlob::Ptr copyBlob(const InferenceEngine::MemoryBlob::Ptr& in, void* ptr);

//
// cvtBlobPrecision
//

void cvtBlobPrecision(const InferenceEngine::MemoryBlob::Ptr& in, const InferenceEngine::MemoryBlob::Ptr& out);

InferenceEngine::MemoryBlob::Ptr toPrecision(const InferenceEngine::MemoryBlob::Ptr& in,
                                             const InferenceEngine::Precision& precision,
                                             const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr,
                                             void* ptr = nullptr);
InferenceEngine::MemoryBlob::Ptr toDefPrecision(const InferenceEngine::MemoryBlob::Ptr& in,
                                                const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr,
                                                void* ptr = nullptr);

inline InferenceEngine::MemoryBlob::Ptr toFP32(const InferenceEngine::MemoryBlob::Ptr& in,
                                               const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr,
                                               void* ptr = nullptr) {
    return toPrecision(in, InferenceEngine::Precision::FP32, allocator, ptr);
}
inline InferenceEngine::MemoryBlob::Ptr toFP16(const InferenceEngine::MemoryBlob::Ptr& in,
                                               const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr,
                                               void* ptr = nullptr) {
    return toPrecision(in, InferenceEngine::Precision::FP16, allocator, ptr);
}

//
// cvtBlobLayout
//

void cvtBlobLayout(const InferenceEngine::MemoryBlob::Ptr& in, const InferenceEngine::MemoryBlob::Ptr& out);

InferenceEngine::MemoryBlob::Ptr toLayout(const InferenceEngine::MemoryBlob::Ptr& in, InferenceEngine::Layout layout,
                                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr,
                                          void* ptr = nullptr);
InferenceEngine::MemoryBlob::Ptr toDefLayout(const InferenceEngine::MemoryBlob::Ptr& in,
                                             const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr,
                                             void* ptr = nullptr);

//
// dumpBlobs
//

void dumpBlobs(const InferenceEngine::BlobMap& blobMap, StringRef dstPath, StringRef blobType);

//
// getMemorySize
//

Byte getMemorySize(const InferenceEngine::TensorDesc& desc);

//
// Check blob Types
//

inline bool isNV12AnyBlob(const InferenceEngine::Blob::CPtr& blob) {
    return blob && blob->is<InferenceEngine::NV12Blob>();
}

inline bool isRemoteBlob(const InferenceEngine::Blob::CPtr& blob) {
    return blob && blob->is<InferenceEngine::RemoteBlob>();
}

inline bool isRemoteNV12Blob(const InferenceEngine::Blob::CPtr& blob) {
    return isNV12AnyBlob(blob) && isRemoteBlob(InferenceEngine::as<InferenceEngine::NV12Blob>(blob)->y()) &&
           isRemoteBlob(InferenceEngine::as<InferenceEngine::NV12Blob>(blob)->uv());
}

inline bool isLocalNV12Blob(const InferenceEngine::Blob::CPtr& blob) {
    return isNV12AnyBlob(blob) && !isRemoteBlob(InferenceEngine::as<InferenceEngine::NV12Blob>(blob)->y()) &&
           !isRemoteBlob(InferenceEngine::as<InferenceEngine::NV12Blob>(blob)->uv());
}

inline bool isRemoteAnyBlob(const InferenceEngine::Blob::CPtr& blob) {
    return (isRemoteBlob(blob) || isRemoteNV12Blob(blob));
}

}  // namespace vpux
