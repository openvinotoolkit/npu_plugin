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

#include "file_reader.h"
#include "ie_compound_blob.h"

#include <precision_utils.h>
#include <fstream>

namespace vpu {

namespace KmbPlugin {

namespace utils {

void fromBinaryFile(std::string input_binary, InferenceEngine::Blob::Ptr blob) {
    std::ifstream in(input_binary, std::ios_base::binary | std::ios_base::ate);

    size_t sizeFile = in.tellg();
    in.seekg(0, std::ios_base::beg);
    size_t count = blob->size();
    if (in.good()) {
        if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
            InferenceEngine::ie_fp16 *blobRawDataFP16 = blob->buffer().as<InferenceEngine::ie_fp16 *>();
            if (sizeFile == count * sizeof(float)) {
                for (size_t i = 0; i < count; i++) {
                    float tmp;
                    in.read(reinterpret_cast<char *>(&tmp), sizeof(float));
                    blobRawDataFP16[i] = InferenceEngine::PrecisionUtils::f32tof16(tmp);
                }
            } else if (sizeFile == count * sizeof(InferenceEngine::ie_fp16)) {
                for (size_t i = 0; i < count; i++) {
                    InferenceEngine::ie_fp16 tmp;
                    in.read(reinterpret_cast<char *>(&tmp), sizeof(InferenceEngine::ie_fp16));
                    blobRawDataFP16[i] = tmp;
                }
            } else {
                THROW_IE_EXCEPTION << "File has invalid size!";
            }
        } else if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
            float *blobRawData = blob->buffer();
            if (sizeFile == count * sizeof(float)) {
                in.read(reinterpret_cast<char *>(blobRawData), count * sizeof(float));
            } else {
                THROW_IE_EXCEPTION << "File has invalid size!";
            }
        } else if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::U8) {
            char *blobRawData = blob->buffer().as<char *>();
            if (sizeFile == count * sizeof(char)) {
                in.read(blobRawData, count * sizeof(char));
            } else {
                THROW_IE_EXCEPTION << "File has invalid size! Blob size: " << count * sizeof(char)
                                   << " file size: " << sizeFile;
            }
        }
    } else {
        THROW_IE_EXCEPTION << "File is not good.";
    }
}

void readNV12FileHelper(const std::string &filePath,
                        size_t sizeToRead,
                        uint8_t *imageData,
                        size_t readOffset) {
    std::ifstream fileReader(filePath, std::ios_base::ate | std::ios_base::binary);
    if (!fileReader.good()) {
        throw std::runtime_error("readNV12FileHelper: failed to open file " + filePath);
    }

    const size_t fileSize = fileReader.tellg();
    if (fileSize - readOffset < sizeToRead) {
        throw std::runtime_error("readNV12FileHelper: size of " + filePath + " is less than expected");
    }
    fileReader.seekg(readOffset, std::ios_base::beg);
    fileReader.read(reinterpret_cast<char *>(imageData), sizeToRead);
    fileReader.close();
}

InferenceEngine::Blob::Ptr fromNV12File(const std::string &filePath,
                                        size_t imageWidth,
                                        size_t imageHeight,
                                        std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> &allocator) {
    const size_t expectedSize = imageWidth * (imageHeight * 3 / 2);
    uint8_t *imageData = reinterpret_cast<uint8_t *>(allocator->allocate(expectedSize));
    readNV12FileHelper(filePath, expectedSize, imageData, 0);

    InferenceEngine::TensorDesc planeY(InferenceEngine::Precision::U8,
        {1, 1, imageHeight, imageWidth}, InferenceEngine::Layout::NHWC);
    InferenceEngine::TensorDesc planeUV(InferenceEngine::Precision::U8,
        {1, 2, imageHeight / 2, imageWidth / 2}, InferenceEngine::Layout::NHWC);
    const size_t offset = imageHeight * imageWidth;

    InferenceEngine::Blob::Ptr blobY = InferenceEngine::make_shared_blob<uint8_t>(planeY, imageData);
    InferenceEngine::Blob::Ptr blobUV = InferenceEngine::make_shared_blob<uint8_t>(planeUV, imageData + offset);

    InferenceEngine::Blob::Ptr nv12Blob = InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(blobY, blobUV);
    return nv12Blob;
}

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
