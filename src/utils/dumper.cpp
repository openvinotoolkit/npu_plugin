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

#include "dumper.hpp"

#include <fstream>

namespace utils {

void dumpBlobHelper(const InferenceEngine::Blob::Ptr& blobPtr, const std::string& dst, const vpu::Logger::Ptr& logger,
    const std::string& blobType) {
    static unsigned dumpCounter = 0;
    std::ostringstream inputFullPath;
    inputFullPath << dst;
    inputFullPath << "/" << blobType << "-dump";
    inputFullPath << dumpCounter++;
    inputFullPath << ".bin";
    logger->info("dumpBlobHelper: dump to file %s", inputFullPath.str());
    std::ofstream dumper(inputFullPath.str(), std::ios_base::binary);
    if (dumper.good()) {
        dumper.write(blobPtr->cbuffer().as<char*>(), blobPtr->byteSize());
    } else {
        logger->warning("dumpBlobHelper: failed to open %s", inputFullPath.str());
    }
    dumper.close();
}

void dumpBlobs(const InferenceEngine::BlobMap& blobMap, const std::string& dstPath, const std::string& blobType,
    const vpu::Logger::Ptr& logger) {
    if (dstPath.empty()) {
        logger->warning("dumpBlobs: destination path is not set.");
        return;
    }
    for (const auto& blob : blobMap) {
        dumpBlobHelper(blob.second, dstPath, logger, blobType);
    }
}

}  // namespace utils
