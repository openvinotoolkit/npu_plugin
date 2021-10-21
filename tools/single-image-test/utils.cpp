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

#include "utils.hpp"

#include <blob_factory.hpp>

#include <fstream>

InferenceEngine::MemoryBlob::Ptr loadBlob(const InferenceEngine::TensorDesc& desc, const std::string& filePath) {
    const auto blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(make_blob_with_precision(desc));
    blob->allocate();

    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    IE_ASSERT(file.is_open()) << "Can't open file " << filePath << " for read";

    const auto blobMem = blob->wmap();
    const auto blobPtr = blobMem.as<char*>();
    file.read(blobPtr, static_cast<std::streamsize>(blob->byteSize()));

    return blob;
}
