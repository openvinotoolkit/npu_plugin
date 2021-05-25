//
// Copyright 2021 Intel Corporation.
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
#include <helper_remote_context.h>
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_helpers/helper_device_name.h"

namespace IE = InferenceEngine;

class HDDL2_InferRequest_PerformanceTests : public testing::Test {
public:
    WorkloadID workloadId = -1;
    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;
    const size_t numberOfTopClassesToCompare = 3;
    const std::string inputNV12Path =
        TestDataHelpers::get_data_path() + "/" + std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "/cat3.yuv";
protected:
    void TearDown() override;
    RemoteMemory_Helper _remoteMemoryHelper;
};

void HDDL2_InferRequest_PerformanceTests::TearDown() { HddlUnite::unregisterWorkloadContext(workloadId); }

// Cover problem: If we have different size of ROI, BlobDesc will created. Avoid this by using original blob size
TEST_F(HDDL2_InferRequest_PerformanceTests, DifferentROISize_NotAffectPerformance) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Set workload context
    WorkloadContext_Helper workloadContextHelper;
    workloadId = workloadContextHelper.getWorkloadId();
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", contextParams);

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load NV12 input
    std::vector<uint8_t> nv12InputBlobMemory;
    size_t nv12Size = inputWidth * inputHeight * 3 / 2;
    nv12InputBlobMemory.resize(nv12Size);
    IE::NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(
        inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());

    // ----- Allocate memory with HddlUnite on device
    auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, nv12Size, inputNV12Blob->y()->cbuffer().as<void*>());

    // ---- Create remote NV12 blob by using already exists remote memory
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

    // Create first ROI blob
    IE::ROI firstRoi {0, 5, 5, inputWidth - 100, inputHeight - 100};
    IE::Blob::Ptr firstROIBlob = remoteNV12BlobPtr->createROI(firstRoi);
    ASSERT_NE(nullptr, firstROIBlob);

    // Second ROI blob with different size
    IE::ROI secondRoi {0, 10, 10, inputWidth - 130, inputHeight - 130};
    IE::Blob::Ptr secondROIBlob = remoteNV12BlobPtr->createROI(secondRoi);
    ASSERT_NE(nullptr, secondROIBlob);

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

    const size_t BLOBS_COUNT = 1000;
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
    const double maxDeviation = 0.06;
    const auto epsilon = std::max(setSingleROITime.count(), setTwoROIInCircleTime.count()) * maxDeviation;

    EXPECT_LE(difference, epsilon);
}
