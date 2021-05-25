//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include "ie_compound_blob.h"
#include "file_reader.h"

namespace NV12Blob_Creator {
    // If data != nullptr, use pre-allocated single memory area, else create 2 different blobs
    InferenceEngine::NV12Blob::Ptr createBlob(const std::size_t width, const std::size_t height, uint8_t* data = nullptr);
    // If data != nullptr, use pre-allocated single memory area, else create 2 different blobs
    InferenceEngine::NV12Blob::Ptr createBlob(const InferenceEngine::TensorDesc& tensorDesc, uint8_t* data = nullptr);
    // If data != nullptr, use pre-allocated single memory area, else create 2 different blobs
    InferenceEngine::NV12Blob::Ptr createFromFile(const std::string& filePath, const std::size_t width, const std::size_t height, uint8_t* data = nullptr);

    void descriptorsFromFrameSize(const size_t width, const size_t height,
                                  InferenceEngine::TensorDesc& uvDesc, InferenceEngine::TensorDesc& yDesc);

    inline InferenceEngine::NV12Blob::Ptr
    createBlob(const std::size_t width, const std::size_t height, uint8_t* data) {
        InferenceEngine::TensorDesc uvDesc;
        InferenceEngine::TensorDesc yDesc;
        NV12Blob_Creator::descriptorsFromFrameSize(width, height, uvDesc, yDesc);
        if (data == nullptr) {
            // Create 2 different blobs
            auto yPlane = InferenceEngine::make_shared_blob<uint8_t>(yDesc);
            yPlane->allocate();

            auto uvPlane = InferenceEngine::make_shared_blob<uint8_t>(uvDesc);
            uvPlane->allocate();

            return InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(yPlane, uvPlane);
        } else {
            // Use pre-allocated single memory area with offsets
            auto yPlane = InferenceEngine::make_shared_blob<uint8_t>(yDesc, data);
            auto uvPlane = InferenceEngine::make_shared_blob<uint8_t>(uvDesc, data + height * width);

            return InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(yPlane, uvPlane);
        }
    }

    inline InferenceEngine::NV12Blob::Ptr createBlob(const InferenceEngine::TensorDesc& tensorDesc, uint8_t* data) {
        if (tensorDesc.getLayout() != InferenceEngine::NCHW && tensorDesc.getLayout() != InferenceEngine::NHWC) {
            IE_THROW() << "Only NCHW / NHWC Layout supported in nv12 blob creator!. Input: "
                               << tensorDesc.getLayout();
        }
        if (tensorDesc.getPrecision() != InferenceEngine::Precision::U8) {
            IE_THROW() << "Only U8 Precision supported in nv12 blob creator!";
        }
        const InferenceEngine::SizeVector& dims = tensorDesc.getDims();
        const int N_index = 0;
        const int C_index = 1;
        const int H_index = 2;
        const int W_index = 3;

        if (dims[N_index] != 1 || dims[C_index] != 3) {
            IE_THROW() << "Only batch 1 and channel == 3 supported for nv12 creator!";
        }

        return createBlob(dims[W_index], dims[H_index], data);
    }

    inline InferenceEngine::NV12Blob::Ptr createFromFile(const std::string& filePath, const std::size_t width, const std::size_t height, uint8_t* data) {
        InferenceEngine::NV12Blob::Ptr nv12Blob = createBlob(width, height, data);
        auto yBlob = nv12Blob->y()->as<InferenceEngine::MemoryBlob>();
        {
            auto yLockMem = yBlob->rmap();
            auto yMem = yLockMem.as<uint8_t*>();
            // TODO [S-28377] call to utils lib
            vpu::KmbPlugin::utils::readNV12FileHelper(filePath, yBlob->size(), yMem, 0);
        }

        auto uvBlob = nv12Blob->uv()->as<InferenceEngine::MemoryBlob>();
        {
            auto uvLockMem = uvBlob->rmap();
            auto uvMem = uvLockMem.as<uint8_t*>();
            // TODO [S-28377] call to utils lib
            vpu::KmbPlugin::utils::readNV12FileHelper(filePath, uvBlob->size(), uvMem, yBlob->size());
        }

        return nv12Blob;
    }

    inline void descriptorsFromFrameSize(const size_t width, const size_t height,
                                         InferenceEngine::TensorDesc& uvDesc, InferenceEngine::TensorDesc& yDesc) {
        uvDesc = {InferenceEngine::Precision::U8, {1, 2, height / 2, width / 2}, InferenceEngine::Layout::NHWC};
        yDesc = {InferenceEngine::Precision::U8, {1, 1, height, width}, InferenceEngine::Layout::NHWC};
    }
}
