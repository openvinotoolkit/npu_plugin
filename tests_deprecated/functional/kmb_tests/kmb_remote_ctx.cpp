//
// Copyright 2020 Intel Corporation.
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

#include <file_reader.h>
#include <gtest/gtest.h>
#include <fstream>

#include <blob_factory.hpp>
#include <test_model/kmb_test_utils.hpp>
#include <vpu_layers_tests.hpp>
#include <ie_compound_blob.h>

#include "vpu/kmb_params.hpp"

using namespace ::testing;
using namespace InferenceEngine;

#if defined(__arm__) || defined(__aarch64__)
TEST_F(vpuLayersTests, remoteCtx) {
    const std::string graphPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";
    const std::string refInputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input.bin";
    const std::string refOutputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output.bin";

    const ParamMap ctxParams = {};
    InferenceEngine::Core ie;
    InferenceEngine::RemoteContext::Ptr contextPtr = ie.CreateContext("KMB", ctxParams);

    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);

    const std::map<std::string, std::string> netParams = {};
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, netParams);
    InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    InferenceEngine::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    const TensorDesc& inputTensorDesc = inputInfoPtr->getTensorDesc();
    const size_t bytesPerElement = inputTensorDesc.getPrecision().size();
    auto binaryMul = [](const size_t& multiplier, const size_t& multiplicand) -> size_t {
        return multiplier * multiplicand;
    };
    auto vpuAllocator = std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    size_t imageSize = std::accumulate(
        inputTensorDesc.getDims().begin(), inputTensorDesc.getDims().end(), bytesPerElement, binaryMul);
    void* virtAddr = vpuAllocator->allocate(imageSize);
    auto remoteMemoryFd = vpuAllocator->getFileDescByVirtAddr(virtAddr);
    InferenceEngine::ParamMap blobParamMap = {
        {InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd},
        {InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE), virtAddr},
    };

    InferenceEngine::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensorDesc, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    const auto userBlob = as<MemoryBlob>(vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inputTensorDesc));
    std::memcpy(virtAddr, userBlob->rmap(), userBlob->byteSize());

    inferRequest.SetBlob(inputName, remoteBlobPtr);
    inferRequest.Infer();

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    const auto outputRefBlob = vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputBlob->getTensorDesc());

    // --- Compare with expected output
    constexpr size_t numberOfTopClassesToCompare = 3;
    ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(outputRefBlob), numberOfTopClassesToCompare));
}

TEST_F(vpuLayersTests, remoteCtxNV12) {
    const std::string graphPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";
    const std::string refInputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-dog-1080x1080-nv12.bin";
    const std::string refOutputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-dog-1080x1080-nv12.bin";

    const ParamMap ctxParams = {};
    InferenceEngine::Core ie;
    InferenceEngine::RemoteContext::Ptr contextPtr = ie.CreateContext("KMB", ctxParams);

    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);

    const std::map<std::string, std::string> netParams = {};
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, netParams);
    InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    InferenceEngine::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> vpuAllocator = std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    InferenceEngine::Blob::Ptr remoteBlobPtr;

    const size_t imWidth = 1080;
    const size_t imHeight = 1080;
    InferenceEngine::Blob::Ptr userBlob =
        vpu::KmbPlugin::utils::fromNV12File(refInputPath, imWidth, imHeight, vpuAllocator);
    NV12Blob::Ptr userNV12blobPtr = as<NV12Blob>(userBlob);
    MemoryBlob::Ptr userYPlane = as<MemoryBlob>(userNV12blobPtr->y());
    MemoryBlob::Ptr userUVPlane = as<MemoryBlob>(userNV12blobPtr->uv());
    auto remoteMemoryFd = vpuAllocator->getFileDescByVirtAddr(userYPlane->rmap().as<void*>());

    TensorDesc ydesc(Precision::U8, { 1, 1, imHeight, imWidth }, Layout::NHWC);
    ParamMap blobParams = {
        { InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd },
        { InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE), userYPlane->rmap().as<KmbHandleParam>() },
    };
    Blob::Ptr y_blob = std::dynamic_pointer_cast<Blob>(contextPtr->CreateBlob(ydesc, blobParams));

    TensorDesc uvdesc(Precision::U8, { 1, 2, imHeight / 2, imWidth / 2 }, Layout::NHWC);
    blobParams[InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE)] = userUVPlane->rmap().as<KmbHandleParam>();
    Blob::Ptr uv_blob = std::dynamic_pointer_cast<Blob>(contextPtr->CreateBlob(uvdesc, blobParams));

    remoteBlobPtr = make_shared_blob<NV12Blob>(y_blob, uv_blob);

    PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
    preprocInfo.setColorFormat(ColorFormat::NV12);

    inferRequest.SetBlob(inputName, remoteBlobPtr, preprocInfo);
    inferRequest.Infer();

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    const auto outputRefBlob = vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputBlob->getTensorDesc());

    // --- Compare with expected output
    constexpr size_t numberOfTopClassesToCompare = 3;
    ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(outputRefBlob), numberOfTopClassesToCompare));
}

struct rectangle {
    size_t x;
    size_t y;
    size_t width;
    size_t height;
};

struct Image {
    size_t width;
    size_t height;
    rectangle rect;
    uint8_t* planes[2];
    int remoteMemoryFd[2];
};

static Blob::Ptr wrapImageToBlob(const Image& image, const RemoteContext::Ptr& contextPtr, bool useOffsets) {
    const size_t& imageWidth = image.width;
    const size_t& imageHeight = image.height;
    TensorDesc planeY(Precision::U8, {1, 1, imageHeight, imageWidth}, Layout::NHWC);
    TensorDesc planeUV(Precision::U8, {1, 2, imageHeight / 2, imageWidth / 2}, Layout::NHWC);
    ROI crop_roi_y({0, image.rect.x, image.rect.y, image.rect.width, image.rect.height});
    ROI crop_roi_uv({0, image.rect.x / 2, image.rect.y / 2, image.rect.width / 2, image.rect.height / 2});

    ParamMap paramsY = { { InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), image.remoteMemoryFd[0] }, };
    if (useOffsets) {
        KmbOffsetParam lumaOffset = 0;
        paramsY[InferenceEngine::KMB_PARAM_KEY(MEM_OFFSET)] = lumaOffset;
    } else {
        paramsY[InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE)] = reinterpret_cast<void*>(image.planes[0]);
    }
    RemoteBlob::Ptr blobY = contextPtr->CreateBlob(planeY, paramsY);
    if (blobY == nullptr) {
        throw std::runtime_error("Failed to create remote blob for Y plane");
    }

    ParamMap paramsUV = { { InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), image.remoteMemoryFd[1] }, };
    if (useOffsets) {
        KmbOffsetParam chromaOffset = imageWidth * imageHeight;
        paramsUV[InferenceEngine::KMB_PARAM_KEY(MEM_OFFSET)] = chromaOffset;
    } else {
        paramsUV[InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE)] = reinterpret_cast<void*>(image.planes[1]);
    }
    RemoteBlob::Ptr blobUV = contextPtr->CreateBlob(planeUV, paramsUV);
    if (blobY == nullptr) {
        throw std::runtime_error("Failed to create remote blob for UV plane");
    }

    Blob::Ptr y_plane_with_roi = make_shared_blob(blobY, crop_roi_y);
    Blob::Ptr uv_plane_with_roi = make_shared_blob(blobUV, crop_roi_uv);

    Blob::Ptr nv12Blob = make_shared_blob<NV12Blob>(y_plane_with_roi, uv_plane_with_roi);
    return nv12Blob;
}

class VpuRemoteCtxTests : public vpuLayersTests, public testing::WithParamInterface<bool> {};

TEST_P(VpuRemoteCtxTests, remoteCtxNV12WithROI) {
    const std::string graphPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";
    const std::string refInputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-dog-1080x1080-nv12.bin";
    const std::string refOutputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-dog-1080x1080-nv12.bin";

    const ParamMap ctxParams = {};
    InferenceEngine::Core ie;
    InferenceEngine::RemoteContext::Ptr contextPtr = ie.CreateContext("KMB", ctxParams);

    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);

    const std::map<std::string, std::string> netParams = {};
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, netParams);
    InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    InferenceEngine::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> vpuAllocator = std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    const size_t imWidth = 1080;
    const size_t imHeight = 1080;
    const size_t lumaSize = imWidth * imHeight;
    const size_t yuvSize = lumaSize * 3 / 2;
    auto imageBaseAddr = reinterpret_cast<uint8_t*>(vpuAllocator->allocate(yuvSize));
    auto remoteMemoryFd = vpuAllocator->getFileDescByVirtAddr(imageBaseAddr);
    std::ifstream inputFileHandle(refInputPath, std::ios_base::binary);
    inputFileHandle.read(reinterpret_cast<char*>(imageBaseAddr), yuvSize);
    inputFileHandle.close();
    Image frame;
    frame.width = imWidth,
    frame.height = imHeight,
    frame.rect.x = 0;
    frame.rect.y = 0;
    frame.rect.width = imWidth;
    frame.rect.height = imHeight;
    frame.planes[0] = imageBaseAddr,
    frame.planes[1] = imageBaseAddr + lumaSize,
    frame.remoteMemoryFd[0] = remoteMemoryFd;
    frame.remoteMemoryFd[1] = remoteMemoryFd;

    bool useOffsets = GetParam();
    InferenceEngine::Blob::Ptr remoteBlobPtr = wrapImageToBlob(frame, contextPtr, useOffsets);

    PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
    preprocInfo.setColorFormat(ColorFormat::NV12);

    inferRequest.SetBlob(inputName, remoteBlobPtr, preprocInfo);
    inferRequest.Infer();

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    const auto outputRefBlob = vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputBlob->getTensorDesc());

    // --- Compare with expected output
    constexpr size_t numberOfTopClassesToCompare = 3;
    ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(outputRefBlob), numberOfTopClassesToCompare));
}

INSTANTIATE_TEST_CASE_P(RemoteCtxWithROI, VpuRemoteCtxTests, ::testing::ValuesIn({true, false}));

TEST_F(vpuLayersTests, incompatibleRemoteCtx) {
    const std::string graphPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";

    InferenceEngine::Core ie;
    InferenceEngine::RemoteContext::Ptr contextPtr = nullptr;

    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);

    const std::map<std::string, std::string> netParams = {};
    InferenceEngine::ExecutableNetwork executableNetwork;
    ASSERT_ANY_THROW(executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, netParams));
}
#endif
