//
// Copyright 2021 Intel Corporation.
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

#include <creators/creator_blob_nv12.h>
#include <gtest/gtest.h>
#include <hddl2_load_network.h>
#include <helper_remote_context.h>
#include <ie_compound_blob.h>
#include <chrono>
#include <fstream>
#include <ie_core.hpp>
#include <tests_common.hpp>
#include "executable_network_factory.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"

namespace IE = InferenceEngine;

class HDDL2_InferRequest_PerformanceTests : public testing::Test {
public:
    WorkloadID workloadId = -1;
    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;
    const size_t nv12Size = inputWidth * inputHeight * 3 / 2;
    const size_t numberOfTopClassesToCompare = 3;

    HddlUnite::RemoteMemory::Ptr allocateRemoteMemory(
            const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize);
    std::string inputNV12Path = TestDataHelpers::get_data_path() + "/" + std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "/cat3.yuv";
protected:
    void TearDown() override;
    HddlUnite::RemoteMemory::Ptr _remoteFrame = nullptr;
};

HddlUnite::RemoteMemory::Ptr HDDL2_InferRequest_PerformanceTests::allocateRemoteMemory(
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

void HDDL2_InferRequest_PerformanceTests::TearDown() { HddlUnite::unregisterWorkloadContext(workloadId); }

// Cover problem: If we have different size of ROI, BlobDesc will created. Avoid this by using original blob size
TEST_F(HDDL2_InferRequest_PerformanceTests, DifferentROISize_NotAffectPerformance) {
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
    IE::NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(inputNV12Path, inputWidth, inputHeight);

    const auto& nv12FrameTensor = IE::TensorDesc(IE::Precision::U8, {1, 1, 1, nv12Size}, IE::Layout::NCHW);
    IE::Blob::Ptr inputRefBlob = make_blob_with_precision(nv12FrameTensor);
    inputRefBlob->allocate();
    const size_t offset = inputNV12Blob->y()->byteSize();
    std::memcpy(IE::as<IE::MemoryBlob>(inputRefBlob)->wmap(), IE::as<IE::MemoryBlob>(inputNV12Blob->y())->rmap(),
                inputNV12Blob->y()->byteSize());
    std::memcpy(IE::as<IE::MemoryBlob>(inputRefBlob)->wmap().as<char*>() + offset,
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
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory},
                                 {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}};

    // Specify input
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

    IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, inputWidth, inputHeight}, IE::Layout::NCHW);
    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensor, blobParamMap);

    // Preprocessing
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // Create first ROI blob
    IE::ROI firstROI{0, 2, 2, 221, 221};
    IE::Blob::Ptr firstROIBlob = remoteBlobPtr->createROI(firstROI);

    // Second ROI blob with different size
    IE::ROI secondROI{0, 2, 2, 211, 211};
    IE::Blob::Ptr secondROIBlob = remoteBlobPtr->createROI(secondROI);

    const size_t BLOBS_COUNT = 500;
    std::cout << "[TEST] Single ROI" << std::endl;
    // Measure time of setting single ROI blob multiple times
    auto start_time = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < BLOBS_COUNT; ++iter) {
        inferRequest.SetBlob(inputName, firstROIBlob, preprocInfo);
        inferRequest.Infer();
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> setSingleROITime = end_time - start_time;

    std::cout << "[TEST] Two ROI" << std::endl;
    // Measure time of setting two ROI in circle
    start_time = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < BLOBS_COUNT; iter+=2) {
        inferRequest.SetBlob(inputName, firstROIBlob, preprocInfo);
        inferRequest.Infer();
        inferRequest.SetBlob(inputName, secondROIBlob, preprocInfo);
        inferRequest.Infer();
    }
    end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> setTwoROIInCircleTime = end_time - start_time;

    std::cout << "[TIME] Set single ROI time: " << setSingleROITime.count() << std::endl;
    std::cout << "[TIME] Set two ROI in circle time: " << setTwoROIInCircleTime.count() << std::endl;

    const auto difference = abs(setTwoROIInCircleTime.count() - setSingleROITime.count());
    const auto epsilon = std::max(setSingleROITime.count(), setTwoROIInCircleTime.count()) * 0.05;
    
    EXPECT_LE(difference, epsilon);
}
