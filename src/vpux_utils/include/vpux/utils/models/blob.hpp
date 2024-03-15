//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

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

std::vector<std::vector<float>> parseBlobAsFP32(const InferenceEngine::BlobMap& outputBlob);

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
// getMemorySize
//

int64_t getMemorySize(const InferenceEngine::TensorDesc& desc);

}  // namespace vpux
