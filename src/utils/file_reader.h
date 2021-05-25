//
// Copyright 2019 Intel Corporation.
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
std::ifstream& skipMagic(std::ifstream& blobStream);

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
