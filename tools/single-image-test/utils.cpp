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
