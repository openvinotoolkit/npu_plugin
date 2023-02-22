//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "file_reader.h"

#include <ie_common.h>
#include <precision_utils.h>

#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <fstream>
#include <ie_icore.hpp>

#include "ie_compound_blob.h"
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vpu {

namespace KmbPlugin {

namespace utils {

size_t getFileSize(std::istream& strm) {
    const size_t streamStart = strm.tellg();
    strm.seekg(0, std::ios_base::end);
    const size_t streamEnd = strm.tellg();
    const size_t bytesAvailable = streamEnd - streamStart;
    strm.seekg(streamStart, std::ios_base::beg);

    return bytesAvailable;
}

InferenceEngine::Blob::Ptr fromBinaryFile(const std::string& input_binary, const InferenceEngine::TensorDesc& desc) {
    std::ifstream in(input_binary, std::ios_base::binary);
    size_t sizeFile = getFileSize(in);

    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(desc);
    blob->allocate();
    size_t count = blob->size();
    if (in.good()) {
        if (desc.getPrecision() == InferenceEngine::Precision::FP16) {
            InferenceEngine::ie_fp16* blobRawDataFP16 = blob->buffer().as<InferenceEngine::ie_fp16*>();
            if (sizeFile == count * sizeof(float)) {
                for (size_t i = 0; i < count; i++) {
                    float tmp;
                    in.read(reinterpret_cast<char*>(&tmp), sizeof(float));
                    blobRawDataFP16[i] = InferenceEngine::PrecisionUtils::f32tof16(tmp);
                }
            } else if (sizeFile == count * sizeof(InferenceEngine::ie_fp16)) {
                for (size_t i = 0; i < count; i++) {
                    InferenceEngine::ie_fp16 tmp;
                    in.read(reinterpret_cast<char*>(&tmp), sizeof(InferenceEngine::ie_fp16));
                    blobRawDataFP16[i] = tmp;
                }
            } else {
                IE_THROW() << "File has invalid size!";
            }
        } else if (desc.getPrecision() == InferenceEngine::Precision::FP32) {
            float* blobRawData = blob->buffer();
            if (sizeFile == count * sizeof(float)) {
                in.read(reinterpret_cast<char*>(blobRawData), count * sizeof(float));
            } else {
                IE_THROW() << "File has invalid size!";
            }
        } else if (desc.getPrecision() == InferenceEngine::Precision::U8) {
            char* blobRawData = blob->buffer().as<char*>();
            if (sizeFile == count * sizeof(char)) {
                in.read(blobRawData, count * sizeof(char));
            } else {
                IE_THROW() << "File has invalid size! Blob size: " << count * sizeof(char)
                           << " file size: " << sizeFile;
            }
        }
    } else {
        IE_THROW() << "File is not good.";
    }
    return blob;
}

void readNV12FileHelper(const std::string& filePath, size_t sizeToRead, uint8_t* imageData, size_t readOffset) {
    std::ifstream fileReader(filePath, std::ios_base::ate | std::ios_base::binary);
    if (!fileReader.good()) {
        throw std::runtime_error("readNV12FileHelper: failed to open file " + filePath);
    }

    const size_t fileSize = fileReader.tellg();
    if (fileSize - readOffset < sizeToRead) {
        throw std::runtime_error("readNV12FileHelper: size of " + filePath + " is less than expected");
    }
    fileReader.seekg(readOffset, std::ios_base::beg);
    fileReader.read(reinterpret_cast<char*>(imageData), sizeToRead);
    fileReader.close();
}

InferenceEngine::Blob::Ptr fromNV12File(const std::string& filePath, size_t imageWidth, size_t imageHeight,
                                        std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>& allocator) {
    const size_t expectedSize = imageWidth * (imageHeight * 3 / 2);
    uint8_t* imageData = reinterpret_cast<uint8_t*>(allocator->allocate(expectedSize));
    readNV12FileHelper(filePath, expectedSize, imageData, 0);

    InferenceEngine::TensorDesc planeY(InferenceEngine::Precision::U8, {1, 1, imageHeight, imageWidth},
                                       InferenceEngine::Layout::NHWC);
    InferenceEngine::TensorDesc planeUV(InferenceEngine::Precision::U8, {1, 2, imageHeight / 2, imageWidth / 2},
                                        InferenceEngine::Layout::NHWC);
    const size_t offset = imageHeight * imageWidth;

    InferenceEngine::Blob::Ptr blobY = InferenceEngine::make_shared_blob<uint8_t>(planeY, imageData);
    InferenceEngine::Blob::Ptr blobUV = InferenceEngine::make_shared_blob<uint8_t>(planeUV, imageData + offset);

    InferenceEngine::Blob::Ptr nv12Blob = InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(blobY, blobUV);
    return nv12Blob;
}

std::istream& skipMagic(std::istream& blobStream) {
    if (!blobStream.good()) {
        IE_THROW(NetworkNotRead);
    }

    using ExportMagic = std::array<char, 4>;
    constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};
    ExportMagic magic = {};

    blobStream.seekg(0, blobStream.beg);
    blobStream.read(magic.data(), magic.size());
    auto exportedWithName = (exportMagic == magic);
    if (exportedWithName) {
        std::string tmp;
        std::getline(blobStream, tmp);
    } else {
        blobStream.seekg(0, blobStream.beg);
    }

    return blobStream;
}

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
