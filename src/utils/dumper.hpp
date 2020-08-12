//
// Copyright 2019-2020 Intel Corporation.
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

#include <ie_blob.h>
#include <ie_common.h>

#include <vpu/utils/logger.hpp>

namespace vpu {

namespace KmbPlugin {

namespace utils {

void dumpBlobHelper(const InferenceEngine::Blob::Ptr& inputBlobPtr, const std::string& dst, const Logger::Ptr& _logger,
    const std::string& blobType);
void dumpBlobs(const InferenceEngine::BlobMap& blobMap, const std::string& dstPath, const std::string& blobType,
    const Logger::Ptr& logger);

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
