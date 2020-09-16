#include "helper_ie_core.h"

#include <format_reader_ptr.h>

#include <blob_factory.hpp>
#include <ie_utils.hpp>
#include <tests_common.hpp>
namespace IE = InferenceEngine;

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

void IE_Core_Helper::checkBBoxOutputs(std::vector<utils::BoundingBox> &actualOutput,
        std::vector<utils::BoundingBox> &refOutput,
        const int imgWidth,
        const int imgHeight,
        const float boxTolerance,
        const float probTolerance) {
    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < refOutput.size(); ++i) {
        const auto& bb = refOutput[i];
        std::cout << i << " : " << bb.idx
                  << " : [("
                  << bb.left << " " << bb.top << "), ("
                  << bb.right << " " << bb.bottom
                  << ")] : "
                  << bb.prob * 100 << "%"
                  << std::endl;
    }

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < actualOutput.size(); ++i) {
        const auto& bb = actualOutput[i];
        std::cout << i << " : " << bb.idx
                  << " : [("
                  << bb.left << " " << bb.top << "), ("
                  << bb.right << " " << bb.bottom
                  << ")] : "
                  << bb.prob * 100 << "%" << std::endl;
    }

    for (const auto& refBB : refOutput) {
        bool found = false;

        float maxBoxError = 0.0f;
        float maxProbError = 0.0f;

        for (const auto& actualBB : actualOutput) {
            if (actualBB.idx != refBB.idx) {
                continue;
            }

            const utils::Box actualBox {
                    actualBB.left / imgWidth,
                    actualBB.top / imgHeight,
                    (actualBB.right - actualBB.left) / imgWidth,
                    (actualBB.bottom - actualBB.top) / imgHeight
            };
            const utils::Box refBox {
                    refBB.left / imgWidth,
                    refBB.top / imgHeight,
                    (refBB.right - refBB.left) / imgWidth,
                    (refBB.bottom - refBB.top) / imgHeight
            };

            const auto boxIntersection = boxIntersectionOverUnion(actualBox, refBox);
            const auto boxError = 1.0f - boxIntersection;
            maxBoxError = std::max(maxBoxError, boxError);

            const auto probError = std::fabs(actualBB.prob - refBB.prob);
            maxProbError = std::max(maxProbError, probError);

            if (boxError > boxTolerance) {
                continue;
            }

            if (probError > probTolerance) {
                continue;
            }

            found = true;
            break;
        }

        EXPECT_TRUE(found)
                            << "maxBoxError=" << maxBoxError << " "
                            << "maxProbError=" << maxProbError;
    }
}
