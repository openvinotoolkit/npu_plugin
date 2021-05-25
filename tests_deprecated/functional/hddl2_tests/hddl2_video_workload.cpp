//
// Copyright 2019 Intel Corporation.
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


#include <ie_blob.h>
#include <blob_factory.hpp>
#include <fstream>

#include <creators/creator_blob_nv12.h>
#include <helper_calc_cpu_ref.h>
#include <helper_ie_core.h>
#include <ie_compound_blob.h>
#include <ie_core.hpp>
#include <tests_common.hpp>
#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include <helper_remote_context.h>
// [Track number: E#12122]
// TODO Remove this header after removing HDDL2 deprecated parameters in future releases
#include "hddl2/hddl2_params.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "executable_network_factory.h"
#include "models/models_constant.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "vpux/utils/IE/blob.hpp"

namespace IE = InferenceEngine;

// [Track number: E#10236]
// TODO Provide a possibility for user to create some default workload context inside VPUXPlugin/HDDL2Backend
// without using low-level HddlUnite API and add tests which are using this default context

// [Track number: S#41888]
// TODO If VideoWorkload samples won't be merged, to get rid of HddlUnite helpers such as RemoteMemory & RemoteContext
// because currently these tests are like the samples

class VideoWorkload_Tests : public ::testing::Test {
public:
    WorkloadID workloadId = -1;
    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;
    const size_t numberOfTopClassesToCompare = 3;

protected:
    void SetUp() override;
    void TearDown() override;
    RemoteMemory_Helper _remoteMemoryHelper;
    WorkloadContext_Helper workloadContextHelper;
};

void VideoWorkload_Tests::SetUp() {
    // ---- Set workload context - we need to get some ID to emulate VAAPI result
    workloadId = workloadContextHelper.getWorkloadId();
}

void VideoWorkload_Tests::TearDown() { HddlUnite::unregisterWorkloadContext(workloadId); }

//------------------------------------------------------------------------------
using VideoWorkload_WithoutPreprocessing = VideoWorkload_Tests;
TEST_F(VideoWorkload_WithoutPreprocessing, precommit_SyncInferenceOneRemoteFrame) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", contextParams);

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load input
    IE::Blob::Ptr inputRefBlob = IE_Core_Helper::loadImage("husky.bmp", inputWidth, inputHeight, IE::NHWC, true);

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, inputRefBlob->size(), inputRefBlob->cbuffer().as<void*>());

    // ---- Create remote blob by using already existing remote memory
    IE::ParamMap blobParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD}};

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // Using value instead of iterator due to a bug in the MSVS 2019 compiler
    auto inputInfo = *executableNetwork.GetInputsInfo().begin();
    const std::string inputName = inputInfo.first;
    IE::InputInfo::CPtr inputInfoPtr = inputInfo.second;

    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputInfoPtr->getTensorDesc(), blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

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

// [Track number: E#12122]
// TODO Remove this test after removing deprecated HDDL2 parameters in future releases
TEST_F(VideoWorkload_WithoutPreprocessing, precommit_SyncInferenceOneRemoteFrame_DeprecatedParams) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", contextParams);

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load input
    IE::Blob::Ptr inputRefBlob = IE_Core_Helper::loadImage("husky.bmp", inputWidth, inputHeight, IE::NHWC, true);

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemory = _remoteMemoryHelper.allocateRemoteMemoryPtr(workloadId, inputRefBlob->size(), inputRefBlob->cbuffer().as<void*>());

    // ---- Create remote blob by using already existing remote memory
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory}};

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // Using value instead of iterator due to a bug in the MSVS 2019 compiler
    auto inputInfo = *executableNetwork.GetInputsInfo().begin();
    const std::string inputName = inputInfo.first;
    IE::InputInfo::CPtr inputInfoPtr = inputInfo.second;

    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputInfoPtr->getTensorDesc(), blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

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

    // ---- Init context map and create context based on it
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", contextParams);

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load input
    IE::Blob::Ptr inputRefBlob = IE_Core_Helper::loadCatImage();

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, inputRefBlob->size(), inputRefBlob->cbuffer().as<void*>());

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already existing remote memory
    IE::ROI roi {0, 10, 10, 100, 100};
    IE::ParamMap blobParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD}};

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

    // ---- Init context map and create context based on it
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", contextParams);

    // ----- Load NV12 input
    std::vector<uint8_t> nv12InputBlobMemory;
    size_t nv12Size = inputWidth * inputHeight * 3 / 2;
    nv12InputBlobMemory.resize(nv12Size);
    IE::NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(
        inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());

    // ----- Allocate memory with HddlUnite on device - emulate VAAPI output
    auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, nv12Size, inputNV12Blob->y()->cbuffer().as<void*>());

    // ---- Create remote NV12 blob by using already existing remote memory
    const size_t yPlanes = 1;
    const size_t uvPlanes = 2;
    IE::ParamMap blobYParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
                                    {IE::VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(0)}};

    IE::TensorDesc inputYTensor = IE::TensorDesc(IE::Precision::U8, {1, yPlanes, inputHeight, inputWidth}, IE::Layout::NHWC);
    IE::RemoteBlob::Ptr remoteYBlobPtr = contextPtr->CreateBlob(inputYTensor, blobYParamMap);
    ASSERT_NE(nullptr, remoteYBlobPtr);

    IE::ParamMap blobUVParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
                                    {IE::VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(inputWidth * inputHeight * yPlanes)}};

    IE::TensorDesc inputUVTensor = IE::TensorDesc(IE::Precision::U8, {1, uvPlanes, inputHeight / 2, inputWidth / 2}, IE::Layout::NHWC);
    IE::RemoteBlob::Ptr remoteUVBlobPtr = contextPtr->CreateBlob(inputUVTensor, blobUVParamMap);
    ASSERT_NE(nullptr, remoteUVBlobPtr);
    IE::NV12Blob::Ptr remoteNV12BlobPtr = IE::make_shared_blob<IE::NV12Blob>(remoteYBlobPtr, remoteUVBlobPtr);
    ASSERT_NE(nullptr, remoteNV12BlobPtr);

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // Specify input
    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // Preprocessing
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set remote NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, remoteNV12BlobPtr, preprocInfo);

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

    // ---- Init context map and create context based on it
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", contextParams);

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load NV12 input
    std::vector<uint8_t> nv12InputBlobMemory;
    nv12InputBlobMemory.resize(inputWidth * inputHeight * 3 / 2);
    IE::NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(
        inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, inputWidth, inputHeight,
        inputWidth, inputHeight, inputNV12Blob->y()->cbuffer().as<void*>(), HddlUnite::eRemoteMemoryFormat::NV12);

    // ---- Create remote NV12 blob by using already existing remote memory
    const size_t yPlanes = 1;
    const size_t uvPlanes = 2;
    IE::ParamMap blobYParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
                                    {IE::VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(0)}};

    IE::TensorDesc inputYTensor = IE::TensorDesc(IE::Precision::U8, {1, yPlanes, inputHeight, inputWidth}, IE::Layout::NHWC);
    IE::RemoteBlob::Ptr remoteYBlobPtr = contextPtr->CreateBlob(inputYTensor, blobYParamMap);
    ASSERT_NE(nullptr, remoteYBlobPtr);

    IE::ParamMap blobUVParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
                                    {IE::VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(inputWidth * inputHeight * yPlanes)}};

    IE::TensorDesc inputUVTensor = IE::TensorDesc(IE::Precision::U8, {1, uvPlanes, inputHeight / 2, inputWidth / 2}, IE::Layout::NHWC);
    IE::RemoteBlob::Ptr remoteUVBlobPtr = contextPtr->CreateBlob(inputUVTensor, blobUVParamMap);
    ASSERT_NE(nullptr, remoteUVBlobPtr);
    IE::NV12Blob::Ptr remoteNV12BlobPtr = IE::make_shared_blob<IE::NV12Blob>(remoteYBlobPtr, remoteUVBlobPtr);
    ASSERT_NE(nullptr, remoteNV12BlobPtr);

    // Create ROI remote blob
    // Fluid backend (CPU) throws the exceptions if roi width/height != blob width/height (for NV12 case)
    // [Track number: S#49686]
    IE::ROI roi {0, 0, 0, inputWidth, inputHeight};
    IE::Blob::Ptr remoteROIBlobPtr =  remoteNV12BlobPtr->createROI(roi);
    ASSERT_NE(nullptr, remoteROIBlobPtr);

    // Create ROI local blob
    IE::Blob::Ptr inputRefBlob = inputNV12Blob->createROI(roi);
    ASSERT_NE(nullptr, inputRefBlob);

    // ---- Import network providing context as input to bind to context
    auto blobContentStream = ExecutableNetworkFactory::getGraphBlob(modelToUse.pathToModel);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobContentStream, contextPtr);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // Specify input
    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

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
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputRefBlob, &preprocInfo);

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

    // ---- Init context map and create context based on it
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", contextParams);

    // ----- Load NV12 input image
    const size_t alignFactor = 256;
    const size_t paddingWidth = alignFactor - inputWidth;
    const size_t yPlanes = 1;
    const size_t uvPlanes = 2;
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

    // ---- Create NV12 local blob - using ROI for strides preprocessing on CPU as reference
    IE::ROI yPlaneRoi {0, 0, 0, inputWidth, inputHeight};
    IE::ROI uvPlaneRoi {0, 0, 0, inputWidth / 2, inputHeight / 2};
    IE::Blob::Ptr yRoiBlob = IE::make_shared_blob(yPlaneInputBlob, yPlaneRoi);
    IE::Blob::Ptr uvRoiBlob = IE::make_shared_blob(uvPlaneInputBlob, uvPlaneRoi);
    IE::Blob::Ptr inputRefBlob = IE::make_shared_blob<IE::NV12Blob>(yRoiBlob, uvRoiBlob);

    // ---- Create NV12 remote blob - using tensor and blocking descriptors for strides preprocessing on VPU
    // ---- First - create repacked data
    const size_t nv12StridesSize = yPlaneStridesSize * 3 / 2;
    std::vector<uint8_t> nv12RepackedData;
    nv12RepackedData.resize(nv12StridesSize);
    const size_t offset = yPlaneStridesSize;
    std::memcpy(nv12RepackedData.data(), IE::as<IE::MemoryBlob>(yRoiBlob)->rmap(), yRoiBlob->byteSize());
    std::memcpy(nv12RepackedData.data() + offset, IE::as<IE::MemoryBlob>(uvRoiBlob)->rmap(), uvRoiBlob->byteSize());

    // ----- Second - allocate memory with HddlUnite on device
    auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, nv12StridesSize, nv12RepackedData.data());

    // ---- Third - create remote NV12 blob by using already existing remote memory
    // Dimensions order: N = 0, C = 1, H = 2, W = 3
    std::vector<size_t> NHWC = {0, 2, 3, 1};
    std::vector<size_t> dimOffsets = {0, 0, 0, 0};

    IE::BlockingDesc inputYBlock = IE::BlockingDesc({1, inputHeight, inputWidth, yPlanes}, NHWC, 0, dimOffsets,
    {(inputWidth + paddingWidth) * inputHeight, inputWidth + paddingWidth, yPlanes, 1});
    IE::BlockingDesc inputUVBlock = IE::BlockingDesc({1, inputHeight / 2, inputWidth / 2, uvPlanes}, NHWC, 0, dimOffsets,
    {(inputWidth + paddingWidth) / 2 * (inputHeight / 2) * uvPlanes, (inputWidth + paddingWidth) / 2 * uvPlanes, uvPlanes, 1});

    IE::TensorDesc inputYTensor = IE::TensorDesc(IE::Precision::U8, {1, yPlanes, inputHeight, inputWidth}, inputYBlock);
    IE::TensorDesc inputUVTensor = IE::TensorDesc(IE::Precision::U8, {1, uvPlanes, inputHeight / 2, inputWidth / 2}, inputUVBlock);

    IE::ParamMap blobYParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
        {IE::VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(0)}};
    IE::RemoteBlob::Ptr remoteYBlobPtr = contextPtr->CreateBlob(inputYTensor, blobYParamMap);
    ASSERT_NE(nullptr, remoteYBlobPtr);

    IE::ParamMap blobUVParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
        {IE::VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(yPlaneStridesSize)}};
    IE::RemoteBlob::Ptr remoteUVBlobPtr = contextPtr->CreateBlob(inputUVTensor, blobUVParamMap);
    ASSERT_NE(nullptr, remoteUVBlobPtr);

    IE::Blob::Ptr inputBlob = IE::make_shared_blob<IE::NV12Blob>(remoteYBlobPtr, remoteUVBlobPtr);
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
            Comparators::compareTopClassesUnordered(vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)), vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)), numberOfTopClassesToCompare));
}

#endif
