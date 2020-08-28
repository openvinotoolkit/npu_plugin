#include "helper_ie_core.h"

#include <format_reader_ptr.h>

#include <blob_factory.hpp>
#include <ie_utils.hpp>
<<<<<<< HEAD
#include <tests_common.hpp>
namespace IE = InferenceEngine;
=======
#include <test_model_repo.hpp>
>>>>>>> Add input layout tests

IE_Core_Helper::IE_Core_Helper()
    : pluginName(
          std::getenv("IE_KMB_TESTS_DEVICE_NAME") != nullptr ? std::getenv("IE_KMB_TESTS_DEVICE_NAME") : "HDDL2") {}

IE::Blob::Ptr IE_Core_Helper::loadCatImage(const IE::Layout& targetImageLayout) {
    // TODO All old blobs RGB based, need to refactor and switch to IR instead
    const bool isBGR = false;
    return loadImage("cat3.bmp", 224, 224, targetImageLayout, isBGR);
}

IE::Blob::Ptr
IE_Core_Helper::loadImage(const std::string &imageName, const size_t width, const size_t height,
                          const IE::Layout targetImageLayout, const bool isBGR) {
    std::ostringstream imageFilePath;
    std::string folder = std::to_string(width) + "x" + std::to_string(height);
    imageFilePath << TestDataHelpers::get_data_path() << "/" << folder << "/" << imageName;

    FormatReader::ReaderPtr reader(imageFilePath.str().c_str());
    IE_ASSERT(reader.get() != nullptr);

    const size_t C = 3;
    const size_t H = height;
    const size_t W = width;

    // CV::Mat - BGR & NHWC
    // Blob for original image
    const IE::SizeVector imageDims = {1, C, H, W};
    const auto tensorDesc = IE::TensorDesc(IE::Precision::FP32, imageDims, IE::Layout::NHWC);
    auto blob = make_blob_with_precision(tensorDesc);
    blob->allocate();

    const auto imagePtr = reader->getData(width, height).get();
    const auto blobPtr = blob->buffer().as<float*>();

    IE_ASSERT(imagePtr != nullptr);
    IE_ASSERT(blobPtr != nullptr);

    if (isBGR) {
        std::copy_n(imagePtr, blob->size(), blobPtr);
    } else {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    blobPtr[c + w * C + h * C * W] = imagePtr[(C - c - 1) + w * C + h * C * W];
                }
            }
        }
    }

    const auto targetPrecision = IE::Precision::U8;
    blob = toPrecision(toLayout(blob, targetImageLayout), targetPrecision);
    return blob;
}
