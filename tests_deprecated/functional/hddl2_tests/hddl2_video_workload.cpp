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


#include <ie_blob.h>
#include <blob_factory.hpp>
#include <fstream>

#include <creators/creator_blob_nv12.h>
#include <helper_calc_cpu_ref.h>
#include <helper_ie_core.h>
#include <ie_compound_blob.h>
#include <ie_core.hpp>
#include <tests_common.hpp>
#include "RemoteMemory.h"
#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "executable_network_factory.h"
#include "models/models_constant.h"

#include <opencv_wraper.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "vpux/utils/IE/blob.hpp"

namespace IE = InferenceEngine;

class VideoWorkload_Tests : public ::testing::Test {
public:
    WorkloadID workloadId = -1;
    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;
    const size_t nv12Size = inputWidth * inputHeight * 3 / 2;
    const size_t numberOfTopClassesToCompare = 3;

    HddlUnite::RemoteMemory::Ptr allocateRemoteMemory(
        const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize);

protected:
    void TearDown() override;
    HddlUnite::RemoteMemory::Ptr _remoteFrame = nullptr;
};

HddlUnite::RemoteMemory::Ptr VideoWorkload_Tests::allocateRemoteMemory(
    const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize) {
    _remoteFrame = std::make_shared<HddlUnite::RemoteMemory> (*context,
        HddlUnite::RemoteMemoryDesc(dataSize, 1, dataSize, 1));

    if (_remoteFrame == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate remote memory.";
    }

    if (_remoteFrame->syncToDevice(data, dataSize) != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to sync memory to device.";
    }
    return _remoteFrame;
}

void VideoWorkload_Tests::TearDown() { HddlUnite::unregisterWorkloadContext(workloadId); }

//------------------------------------------------------------------------------
using VideoWorkload_WithoutPreprocessing = VideoWorkload_Tests;
TEST_F(VideoWorkload_WithoutPreprocessing, precommit_SyncInferenceOneRemoteFrame) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load input
    IE::Blob::Ptr inputRefBlob = IE_Core_Helper::loadImage("husky.bmp", modelToUse.width, modelToUse.height, IE::NHWC, true);

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemory = allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", paramMap);

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already exists remote memory
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory}};

    // Using value instead of iterator due to a bug in the MSVS 2019 compiler
    auto inputInfo = *executableNetwork.GetInputsInfo().begin();
    const std::string inputName = inputInfo.first;
    IE::InputInfo::CPtr inputInfoPtr = inputInfo.second;

    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputInfoPtr->getTensorDesc(), blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // ---- Set remote blob as input for infer request
    inferRequest.SetBlob(inputName, remoteBlobPtr);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputRefBlob);

    // --- Compare with expected output
    ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
                        vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                        vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                        numberOfTopClassesToCompare));
}

TEST_F(VideoWorkload_WithoutPreprocessing, precommit_SyncInferenceOneRemoteFrameROI_Unsupported) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load input
    IE::Blob::Ptr inputRefBlob = IE_Core_Helper::loadCatImage();

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemory = allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", paramMap);

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already exists remote memory
    IE::ROI roi {0, 10, 10, 100, 100};
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory}};

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputInfoPtr->getTensorDesc(), blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);
    IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));
    ASSERT_NE(nullptr, remoteROIBlobPtr);

    // ---- Set remote blob as input for infer request
    inferRequest.SetBlob(inputName, remoteROIBlobPtr);

    // ---- Run the request synchronously
    ASSERT_ANY_THROW(inferRequest.Infer());  //  expected 'preprocess only support NV12 format'
}

//------------------------------------------------------------------------------
class VideoWorkload_WithPreprocessing : public VideoWorkload_Tests {
public:
    std::string inputNV12Path;

protected:
    void SetUp() override;
};

void VideoWorkload_WithPreprocessing::SetUp() {
    VideoWorkload_Tests::SetUp();
    inputNV12Path = TestDataHelpers::get_data_path() + "/" + std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "/cat3.yuv";
}

TEST_F(VideoWorkload_WithPreprocessing, precommit_onOneRemoteFrame) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load NV12 input
    IE::NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(
        inputNV12Path, inputWidth, inputHeight);

    const auto& nv12FrameTensor =
        IE::TensorDesc(IE::Precision::U8, {1, 1, 1, nv12Size}, IE::Layout::NCHW);
    IE::Blob::Ptr inputRefBlob = make_blob_with_precision(nv12FrameTensor);
    inputRefBlob->allocate();
    const size_t offset = inputNV12Blob->y()->byteSize();
    std::memcpy(IE::as<IE::MemoryBlob>(inputRefBlob)->wmap(),
        IE::as<IE::MemoryBlob>(inputNV12Blob->y())->rmap(), inputNV12Blob->y()->byteSize());
    std::memcpy(IE::as<IE::MemoryBlob>(inputRefBlob)->wmap().as<char *>() + offset,
        IE::as<IE::MemoryBlob>(inputNV12Blob->uv())->rmap(), inputNV12Blob->uv()->byteSize());

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemory = allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", paramMap);

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already exists remote memory and specify color format of it
    IE::ParamMap blobParamMap = {
        {IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory}, {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}};

    // Specify input
    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, inputHeight, inputWidth}, IE::Layout::NCHW);
    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensor, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // Preprocessing
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set remote NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, remoteBlobPtr, preprocInfo);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputNV12Blob, &preprocInfo);

    ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
                        vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                        vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                        numberOfTopClassesToCompare));
}

TEST_F(VideoWorkload_WithPreprocessing, precommit_onOneRemoteFrameROI) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load NV12 input
    IE::NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(
        inputNV12Path, inputWidth, inputHeight);

    const auto& nv12FrameTensor =
        IE::TensorDesc(IE::Precision::U8, {1, 1, 1, nv12Size}, IE::Layout::NCHW);
    IE::Blob::Ptr inputRefBlob = make_blob_with_precision(nv12FrameTensor);
    inputRefBlob->allocate();
    const size_t offset = inputNV12Blob->y()->byteSize();
    std::memcpy(IE::as<IE::MemoryBlob>(inputRefBlob)->wmap(),
        IE::as<IE::MemoryBlob>(inputNV12Blob->y())->rmap(), inputNV12Blob->y()->byteSize());
    std::memcpy(IE::as<IE::MemoryBlob>(inputRefBlob)->wmap().as<char *>() + offset,
        IE::as<IE::MemoryBlob>(inputNV12Blob->uv())->rmap(), inputNV12Blob->uv()->byteSize());

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemory = allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", paramMap);

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    IE::ROI roi {0, 2, 2, 221, 221};

    // ---- Create remote blob by using already exists remote memory and specify color format of it
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory},
        {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}};

    // Specify input
    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, inputHeight, inputWidth}, IE::Layout::NCHW);
    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensor, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);
    IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));
    ASSERT_NE(nullptr, remoteROIBlobPtr);

    // Preprocessing
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set remote NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, remoteROIBlobPtr, preprocInfo);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputNV12Blob, &preprocInfo);

    ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
                        vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                        vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                        numberOfTopClassesToCompare));
}

#ifdef USE_OPENCV

// Case: remote frame with 256B alignment
TEST_F(VideoWorkload_WithPreprocessing, precommit_onOneRemoteFrameWithStrides) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());
    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", paramMap);

    // ----- Load NV12 input image
    const size_t alignFactor = 256;
    const size_t paddingWidth = alignFactor - inputWidth;
    const size_t yPlanes = 1;
    const size_t uvPlanes = 2;
    const size_t allPlanes = 3;
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(
                                            inputNV12Path, inputWidth, inputHeight);

    // Create OpenCV image from Y plane
    std::vector <uint8_t> yPlaneOrigData;
    {
    const auto lockedMemory = IE::as<IE::MemoryBlob>(nv12InputBlob->y())->rmap();
    const auto data = lockedMemory.as<uint8_t*>();
    yPlaneOrigData.assign(data, data + nv12InputBlob->y()->byteSize());
    }
    cv::Mat originalImage = cv::Mat(inputHeight, inputWidth, CV_8UC1, yPlaneOrigData.data());

    // Add padding to Y plane at right side
    const size_t yPlaneStridesSize = (inputWidth + paddingWidth) * inputHeight * yPlanes;
    std::vector<uint8_t> yPlaneDstData(yPlaneStridesSize);
    cv::Mat dstImg(inputHeight, inputWidth + paddingWidth, CV_8UC1, yPlaneDstData.data());
    cv::copyMakeBorder(originalImage, dstImg, 0, 0, 0, paddingWidth, cv::BORDER_WRAP);

    // Create blob for Y plane with padding
    IE::SizeVector yPlaneDims {1, yPlanes, inputHeight, inputWidth + paddingWidth};
    IE::TensorDesc yPlaneTensorDesc(IE::Precision::U8, yPlaneDims, IE::Layout::NHWC);
    IE::Blob::Ptr yPlaneInputBlob = IE::make_shared_blob<uint8_t>(yPlaneTensorDesc);
    yPlaneInputBlob->allocate();
    {
        auto blobPtr = yPlaneInputBlob->buffer().as<uint8_t*>();
        std::copy_n(yPlaneDstData.data(), yPlaneInputBlob->size(), blobPtr);
    }

    // Make fictive gray-scale uvPlane with padding
    IE::SizeVector uvPlaneDims {1, uvPlanes, inputHeight / 2, (inputWidth + paddingWidth) / 2};
    IE::TensorDesc uvPlaneTensorDesc(IE::Precision::U8, uvPlaneDims, IE::Layout::NHWC);
    const int64_t grayConst = 0x80;
    IE::Blob::Ptr uvPlaneInputBlob = vpux::makeSplatBlob(uvPlaneTensorDesc, grayConst);

    // ---- Create NV12 local blob with ROI for strides preprocessing on CPU as reference
    IE::ROI yPlaneRoi {0, 0, 0, inputWidth, inputHeight};
    IE::ROI uvPlaneRoi {0, 0, 0, inputWidth / 2, inputHeight / 2};
    IE::Blob::Ptr yRoiBlob = IE::make_shared_blob(yPlaneInputBlob, yPlaneRoi);
    IE::Blob::Ptr uvRoiBlob = IE::make_shared_blob(uvPlaneInputBlob, uvPlaneRoi);
    IE::Blob::Ptr inputRefBlob = IE::make_shared_blob<IE::NV12Blob>(yRoiBlob, uvRoiBlob);

    // ---- Create NV12 repacked remote blob with ROI for strides preprocessing on VPU
    // ---- First - create repacked local blob
    const size_t nv12StridesSize = yPlaneStridesSize * 3 / 2;
    const auto& nv12RepackedTensor =
        IE::TensorDesc(IE::Precision::U8, {1, 1, 1, nv12StridesSize}, IE::Layout::NCHW);
    IE::Blob::Ptr nv12RepackedBlob = make_blob_with_precision(nv12RepackedTensor);
    nv12RepackedBlob->allocate();
    const size_t offset = yPlaneStridesSize;
    std::memcpy(IE::as<IE::MemoryBlob>(nv12RepackedBlob)->wmap(),
        IE::as<IE::MemoryBlob>(yRoiBlob)->rmap(), yRoiBlob->byteSize());
    std::memcpy(IE::as<IE::MemoryBlob>(nv12RepackedBlob)->wmap().as<char *>() + offset,
        IE::as<IE::MemoryBlob>(uvRoiBlob)->rmap(), uvRoiBlob->byteSize());

    // ---- Second - allocate remote memory and bind it to blob data
    auto remoteMemory = allocateRemoteMemory(context, nv12RepackedBlob->buffer().as<void*>(), nv12RepackedBlob->size());
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory},
        {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}};

    // ---- Third - create remote blob and add ROI for strides preprocessing on VPU
    IE::ROI remoteRoi {0, 0, 0, inputWidth, inputHeight};
    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, allPlanes, inputHeight, inputWidth + paddingWidth}, IE::Layout::NCHW);
    IE::RemoteBlob::Ptr remoteBlob = contextPtr->CreateBlob(inputTensor, blobParamMap);
    ASSERT_NE(nullptr, remoteBlob);
    IE::Blob::Ptr inputBlob = remoteBlob->createROI(remoteRoi);
    ASSERT_NE(nullptr, inputBlob);

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // ---- Preprocessing
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputBlobName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set NV12 blob with preprocessing information
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob, preprocInfo));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputRefBlob,  &preprocInfo);

    // --- Compare with reference
    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
            Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));
}

#endif
