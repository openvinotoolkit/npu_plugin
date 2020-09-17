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

#include <core_api.h>
#include <ie_blob.h>

#include <blob_factory.hpp>
#include <chrono>
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

class Performance_Tests : public CoreAPI_Tests {
public:
    using Time = std::chrono::high_resolution_clock::time_point;
    Time (&Now)() = std::chrono::high_resolution_clock::now;

    WorkloadID workloadId = -1;

    const int numberOfIterations = 100;
    std::string graphPath;
    std::string refInputPath;
    std::string refOutputPath;

    const size_t numberOfTopClassesToCompare = 3;

    HddlUnite::SMM::RemoteMemory::Ptr allocateRemoteMemory(
        const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t& dataSize);

protected:
    void SetUp() override;
    void TearDown() override;

    HddlUnite::SMM::RemoteMemory::Ptr _remoteFrame = nullptr;
};

HddlUnite::SMM::RemoteMemory::Ptr Performance_Tests::allocateRemoteMemory(
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

void Performance_Tests::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50.nv12_1080Input;
    refOutputPath = PrecompiledResNet_Helper::resnet50.nv12_1080Output;
}

void Performance_Tests::TearDown() { HddlUnite::unregisterWorkloadContext(workloadId); }

// [Track number: S#31183]
TEST_F(Performance_Tests, DISABLED_Resnet50_DPU_Blob_WithPreprocessing) {
    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load binary input
    const auto& nv12FrameTensor =
        IE::TensorDesc(IE::Precision::U8, {1, 1, 1, 1749600}, IE::Layout::NCHW);

    IE::Blob::Ptr inputRefBlob;
    ASSERT_NO_THROW(inputRefBlob = vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, nv12FrameTensor));

    // ----- Allocate memory with HddlUnite on device
    HddlUnite::SMM::RemoteMemory::Ptr remoteMemory =
        allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", paramMap);

    // ---- Import network providing context as input to bind to context
    const std::string& modelPath = graphPath;

    std::filebuf blobFile;
    if (!blobFile.open(modelPath, std::ios::in | std::ios::binary)) {
        THROW_IE_EXCEPTION << "Could not open file: " << modelPath;
    }
    std::istream graphBlob(&blobFile);

    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);
    blobFile.close();

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    Time start_sync;
    Time end_sync;

    start_sync = Now();
    for (int i = 0; i < numberOfIterations; ++i) {
        IE::ROI roi {0, 2, 2, 1077, 1077};

        // ---- Create remote blob by using already exists remote memory and specify color format of it
        IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory},
            {IE::HDDL2_PARAM_KEY(COLOR_FORMAT), IE::ColorFormat::NV12}};

        // Specify input
        auto inputsInfo = executableNetwork.GetInputsInfo();
        const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

        IE::TensorDesc inputTensor = IE::TensorDesc(IE::Precision::U8, {1, 3, 1080, 1080}, IE::Layout::NCHW);
        IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensor, blobParamMap);
        IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));

        // Since it 228x228 image on 224x224 network, resize preprocessing also required
        IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
        preprocInfo.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        preprocInfo.setColorFormat(IE::ColorFormat::NV12);
        // ---- Set remote NV12 blob with preprocessing information
        inferRequest.SetBlob(inputName, remoteROIBlobPtr, preprocInfo);

        // ---- Run the request synchronously
        inferRequest.Infer();
    }
    end_sync = Now();

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr outputRefBlob;
    ASSERT_NO_THROW(outputRefBlob = vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputBlob->getTensorDesc()));

    // --- Compare with expected output
    ASSERT_TRUE(outputBlob->byteSize() == outputRefBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == IE::Precision::U8);
    ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(outputBlob, outputRefBlob, numberOfTopClassesToCompare));
    auto elapsedSync = std::chrono::duration_cast<std::chrono::milliseconds>(end_sync - start_sync);
    auto executionTimeMs = elapsedSync.count();

    std::cout << "Execution inference (ms): " << executionTimeMs << " on " << numberOfIterations << " iterations"
              << std::endl;
    std::cout << "One frame execution (ms): " << executionTimeMs / numberOfIterations << std::endl;
    const auto inferencePerSeconds = 1000 / ((float)executionTimeMs / numberOfIterations);
    std::cout << "Inference per seconds (fps): " << inferencePerSeconds << std::endl;

    // TODO Here we should compare inferAsync time execution, not full pipeline
    ASSERT_GT(inferencePerSeconds, 24);
}
