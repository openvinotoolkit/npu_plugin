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

#pragma once

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <ie_blob.h>

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

}  // namespace vpux
