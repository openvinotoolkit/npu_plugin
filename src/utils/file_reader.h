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

#include <ie_blob.h>

#include <blob_factory.hpp>
#include <string>

#include "allocators.hpp"

namespace vpu {

namespace KmbPlugin {

namespace utils {

size_t getFileSize(std::istream& strm);
InferenceEngine::Blob::Ptr fromBinaryFile(const std::string& input_binary, const InferenceEngine::TensorDesc& desc);
void readNV12FileHelper(const std::string& filePath, size_t sizeToRead, uint8_t* imageData, size_t readOffset);
InferenceEngine::Blob::Ptr fromNV12File(
    const std::string& filePath, size_t imageWidth, size_t imageHeight, std::shared_ptr<VPUAllocator>& allocator);

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
