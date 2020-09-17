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
#include <hddl2_plugin_config.hpp>

#include <creators/creator_blob_nv12.h>
#include <tests_common.hpp>
#include <helper_ie_core.h>
#include <ie_core.hpp>
#include <ie_compound_blob.h>
#include "RemoteMemory.h"
#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "ie_core.hpp"
#include "ie_utils.hpp"
#include <helper_calc_cpu_ref.h>
#include "models/precompiled_resnet.h"
#include <vpu/utils/ie_helpers.hpp>

namespace IE = InferenceEngine;

class VideoWorkload_Tests : public ::testing::Test {
public:
    WorkloadID workloadId = -1;

    std::string graphPath;
    std::string modelPath;

    const size_t inputWidth = 224;
    const size_t inputHeight = 224;
    const size_t nv12Size = inputWidth * inputHeight * 3 / 2;
    const size_t numberOfTopClassesToCompare = 4;

    HddlUnite::SMM::RemoteMemory::Ptr allocateRemoteMemory(
        const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize);

protected:
    void SetUp() override;
    void TearDown() override;
    HddlUnite::SMM::RemoteMemory::Ptr _remoteFrame = nullptr;
};

HddlUnite::SMM::RemoteMemory::Ptr VideoWorkload_Tests::allocateRemoteMemory(
    const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize) {
    _remoteFrame = HddlUnite::SMM::allocate(*context, dataSize);

    if (_remoteFrame == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate remote memory.";
    }

    if (_remoteFrame->syncToDevice(data, dataSize) != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to sync memory to device.";
    }
    return _remoteFrame;
}

void VideoWorkload_Tests::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
    modelPath = PrecompiledResNet_Helper::resnet50.modelPath;
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
    IE::Blob::Ptr inputRefBlob = IE_Core_Helper::loadCatImage(IE::Layout::NCHW);

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemory = allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);
    blobFile.close();

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
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputRefBlob);

    // --- Compare with expected output
    ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
        toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
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
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, "HDDL2");
    blobFile.close();

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
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);
    blobFile.close();

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Create remote blob by using already exists remote memory and specify color format of it
    IE::ParamMap blobParamMap = {
        {IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory}, {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}};

    // Specify input
    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, inputWidth, inputHeight}, IE::Layout::NCHW);
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
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputNV12Blob, &preprocInfo);

    ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
        toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
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
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);
    blobFile.close();

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

    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, inputWidth, inputHeight}, IE::Layout::NCHW);
    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensor, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);
    IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));
    ASSERT_NE(nullptr, remoteROIBlobPtr);    

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
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputNV12Blob, &preprocInfo);

    ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
        toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}
