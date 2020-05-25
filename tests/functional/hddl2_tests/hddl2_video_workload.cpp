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

#include "RemoteMemory.h"
#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "ie_compound_blob.h"
#include "ie_core.hpp"
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

using RemoteMemoryFD = uint64_t;
class VideoWorkload_Tests : public ::testing::Test {
public:
    WorkloadID workloadId = -1;

    std::string graphPath;
    std::string refInputPath;
    std::string refOutputPath;

    const size_t numberOfTopClassesToCompare = 5;
    const size_t reducedNumberOfTopClassesToCompare = 3;
    RemoteMemoryFD allocateRemoteMemory(
        const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize);

protected:
    void SetUp() override;
    void TearDown() override;
    HddlUnite::SMM::RemoteMemory::Ptr _remoteFrame = nullptr;
};

RemoteMemoryFD VideoWorkload_Tests::allocateRemoteMemory(
    const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize) {
    _remoteFrame = HddlUnite::SMM::allocate(*context, dataSize);

    if (_remoteFrame == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate remote memory.";
    }

    if (_remoteFrame->syncToDevice(data, dataSize) != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to sync memory to device.";
    }
    return _remoteFrame->getDmaBufFd();
}

void VideoWorkload_Tests::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50_dpu.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50_dpu.inputPath;
    refOutputPath = PrecompiledResNet_Helper::resnet50_dpu.outputPath;
}

void VideoWorkload_Tests::TearDown() { HddlUnite::unregisterWorkloadContext(workloadId); }

//------------------------------------------------------------------------------
using VideoWorkload_WithoutPreprocessing = VideoWorkload_Tests;
TEST_F(VideoWorkload_WithoutPreprocessing, SyncInferenceOneRemoteFrame) {
    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load binary input
    const auto& inputTensor = PrecompiledResNet_Helper::resnet50_dpu_tensors.inputTensor;
    auto inputRefBlob = make_blob_with_precision(inputTensor);
    inputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inputRefBlob));

    // ----- Allocate memory with HddlUnite on device
    RemoteMemoryFD remoteMemoryFd =
        allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    const std::string& modelPath = graphPath;

    std::filebuf blobFile;
    if (!blobFile.open(modelPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << modelPath;
    }
    std::istream graphBlob(&blobFile);

    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already exists fd
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd}};

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

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
    auto outputRefBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    outputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputRefBlob));

    // --- Compare with expected output
    ASSERT_TRUE(outputBlob->byteSize() == outputRefBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == IE::Precision::U8);
    ASSERT_NO_THROW(Comparators::compareTopClasses(outputBlob, outputRefBlob, numberOfTopClassesToCompare));
}

//------------------------------------------------------------------------------
TEST_F(VideoWorkload_WithoutPreprocessing, SyncInferenceOneRemoteFrameROI_Unsupported) {
    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load binary input
    const auto& inputTensor = PrecompiledResNet_Helper::resnet50_dpu_tensors.inputTensor;
    auto inputRefBlob = make_blob_with_precision(inputTensor);
    inputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inputRefBlob));

    // ----- Allocate memory with HddlUnite on device
    RemoteMemoryFD remoteMemoryFd =
        allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    const std::string& modelPath = graphPath;

    std::filebuf blobFile;
    if (!blobFile.open(modelPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << modelPath;
    }
    std::istream graphBlob(&blobFile);

    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already exists fd
    IE::ROI roi {0, 10, 10, 100, 100};
    IE::ParamMap blobParamMap = {
        {IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd}, {IE::HDDL2_PARAM_KEY(ROI), roi}};

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputInfoPtr->getTensorDesc(), blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // ---- Set remote blob as input for infer request
    inferRequest.SetBlob(inputName, remoteBlobPtr);

    // ---- Run the request synchronously
    ASSERT_ANY_THROW(inferRequest.Infer());  //  expected 'preprocess only support NV12 format'
}

//------------------------------------------------------------------------------
class VideoWorkload_WithPreprocessing : public VideoWorkload_Tests {
protected:
    void SetUp() override;
};

void VideoWorkload_WithPreprocessing::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50_dpu.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50_dpu.nv12Input;
    refOutputPath = PrecompiledResNet_Helper::resnet50_dpu.nv12Output;
}

TEST_F(VideoWorkload_WithPreprocessing, onOneRemoteFrame) {
    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load binary input
    const auto& nv12FrameTensor =
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, {1, 1, 1, 77976}, InferenceEngine::Layout::NCHW);

    auto inputRefBlob = make_blob_with_precision(nv12FrameTensor);
    inputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inputRefBlob));

    // ----- Allocate memory with HddlUnite on device
    RemoteMemoryFD remoteMemoryFd =
        allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    const std::string& modelPath = graphPath;

    std::filebuf blobFile;
    if (!blobFile.open(modelPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << modelPath;
    }
    std::istream graphBlob(&blobFile);

    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already exists fd and specify color format of it
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd},
        {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}};

    // Specify input
    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, 228, 228}, IE::Layout::NCHW);
    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensor, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // Since it 228x228 image on 224x224 network, resize preprocessing also required
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(IE::RESIZE_BILINEAR);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set remote NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, remoteBlobPtr, preprocInfo);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto outputRefBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    outputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputRefBlob));

    // --- Compare with expected output
    ASSERT_TRUE(outputBlob->byteSize() == outputRefBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == IE::Precision::U8);
    ASSERT_NO_THROW(Comparators::compareTopClasses(outputBlob, outputRefBlob, numberOfTopClassesToCompare));
}

TEST_F(VideoWorkload_WithPreprocessing, onOneRemoteFrameROI) {
    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load binary input
    const auto& nv12FrameTensor =
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, {1, 1, 1, 77976}, InferenceEngine::Layout::NCHW);

    auto inputRefBlob = make_blob_with_precision(nv12FrameTensor);
    inputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inputRefBlob));

    // ----- Allocate memory with HddlUnite on device
    RemoteMemoryFD remoteMemoryFd =
        allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    const std::string& modelPath = graphPath;

    std::filebuf blobFile;
    if (!blobFile.open(modelPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << modelPath;
    }
    std::istream graphBlob(&blobFile);

    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    IE::ROI roi {0, 2, 2, 221, 221};

    // ---- Create remote blob by using already exists fd and specify color format of it
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd},
        {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}, {IE::HDDL2_PARAM_KEY(ROI), roi}};

    // Specify input
    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, 228, 228}, IE::Layout::NCHW);
    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensor, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // Since it 228x228 image on 224x224 network, resize preprocessing also required
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(IE::RESIZE_BILINEAR);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set remote NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, remoteBlobPtr, preprocInfo);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto outputRefBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    outputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputRefBlob));

    // --- Compare with expected output
    ASSERT_TRUE(outputBlob->byteSize() == outputRefBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == IE::Precision::U8);
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(outputBlob, outputRefBlob, reducedNumberOfTopClassesToCompare));
}
