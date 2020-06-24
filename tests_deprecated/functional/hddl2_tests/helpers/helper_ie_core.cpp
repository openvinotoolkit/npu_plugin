#include "helper_ie_core.h"

#include <format_reader_ptr.h>

#include <blob_factory.hpp>
#include <ie_utils.hpp>
#include <tests_common.hpp>

IE_Core_Helper::IE_Core_Helper()
    : pluginName(
          std::getenv("IE_KMB_TESTS_DEVICE_NAME") != nullptr ? std::getenv("IE_KMB_TESTS_DEVICE_NAME") : "HDDL2") {}

InferenceEngine::Blob::Ptr IE_Core_Helper::loadCatImage(const InferenceEngine::Layout& targetImageLayout) {
    std::string catImagePath = "224x224/cat3.bmp";
    std::ostringstream imageFilePath;
    imageFilePath << TestDataHelpers::get_data_path() << "/" << catImagePath;

    FormatReader::ReaderPtr reader(imageFilePath.str().c_str());
    IE_ASSERT(reader.get() != nullptr);

    const size_t C = 3;
    const size_t H = 224;
    const size_t W = 224;

    // CV::Mat BGR & NHWC
    // In NCHW format
    const InferenceEngine::SizeVector imageDims = {1, C, H, W};
    const auto tensorDesc =
        InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, imageDims, InferenceEngine::Layout::NHWC);

    const auto blob = make_blob_with_precision(tensorDesc);
    blob->allocate();

    const auto imagePtr = reader->getData(W, H).get();
    const auto blobPtr = blob->buffer().as<float*>();

    IE_ASSERT(imagePtr != nullptr);
    IE_ASSERT(blobPtr != nullptr);

    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            for (size_t c = 0; c < C; ++c) {
                blobPtr[c + w * C + h * C * W] = imagePtr[(C - c - 1) + w * C + h * C * W];
            }
        }
    }
    const InferenceEngine::TensorDesc targetDesc(InferenceEngine::Precision::U8, targetImageLayout);

    return toPrecision(toLayout(blob, targetDesc.getLayout()), targetDesc.getPrecision());
};
