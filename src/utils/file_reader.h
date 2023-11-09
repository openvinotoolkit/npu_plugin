//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <ie_blob.h>

#include <blob_factory.hpp>
#include <fstream>
#include <string>

#include "allocators.hpp"

namespace vpu {

namespace KmbPlugin {

namespace utils {

size_t getFileSize(std::istream& strm);
InferenceEngine::Blob::Ptr fromBinaryFile(const std::string& input_binary, const InferenceEngine::TensorDesc& desc);
void readNV12FileHelper(const std::string& filePath, size_t sizeToRead, uint8_t* imageData, size_t readOffset);
InferenceEngine::Blob::Ptr fromNV12File(const std::string& filePath, size_t imageWidth, size_t imageHeight,
                                        std::shared_ptr<VPUAllocator>& allocator);
std::istream& skipMagic(std::istream& blobStream);

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
