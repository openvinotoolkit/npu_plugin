#include "helper_ie_core.h"

IE_Core_Helper::IE_Core_Helper()
    : pluginName(
          std::getenv("IE_KMB_TESTS_DEVICE_NAME") != nullptr ? std::getenv("IE_KMB_TESTS_DEVICE_NAME") : "HDDL2") {}

void IE_Core_Helper::printRawBlob(
    const InferenceEngine::Blob::Ptr& blob, const size_t& sizeToPrint, const std::string& blobName) {
    if (blob->size() < sizeToPrint) {
        THROW_IE_EXCEPTION << "Blob size is smaller then required size to print.\n"
                              "Blob size: "
                           << blob->size() << " sizeToPrint: " << sizeToPrint;
    }
    if (blob->getTensorDesc().getPrecision() != InferenceEngine::Precision::U8) {
        THROW_IE_EXCEPTION << "Unsupported precision";
    }

    auto mBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    auto mappedMemory = mBlob->rmap();
    uint8_t* data = mappedMemory.as<uint8_t*>();
    std::cout << "uint8_t output raw data" << (blobName.empty() ? "" : " for " + blobName) << "\t: " << std::hex;
    for (size_t i = 0; i < sizeToPrint; ++i) {
        std::cout << unsigned(data[i]) << " ";
    }
    std::cout << std::endl;
}
