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

#include <blob_factory.hpp>
#include <fstream>

#include "RemoteMemory.h"
#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "ie_core.hpp"
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

using RemoteMemoryFD = uint64_t;
class VideoWorkload_Tests : public ::testing::Test {
public:
    const size_t numberOfTopClassesToCompare = 5;
    RemoteMemoryFD allocateRemoteMemory(
        const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t dataSize);

protected:
    HddlUnite::SMM::RemoteMemory::Ptr _remoteFrame = nullptr;
};

RemoteMemoryFD VideoWorkload_Tests::allocateRemoteMemory(
    const HddlUnite::WorkloadContext::Ptr& context, const void* data, const size_t dataSize) {
    _remoteFrame = HddlUnite::SMM::allocate(*context, dataSize);

    if (_remoteFrame == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate remote memory.";
    }

    if (_remoteFrame->syncToDevice(data, dataSize) != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to sync memory to device.";
    }
    return _remoteFrame->getDmaBufFd();
}

//------------------------------------------------------------------------------
using VideoWorkload_WithoutPreprocessing = VideoWorkload_Tests;
// [Track number: S#28336]
TEST_F(VideoWorkload_WithoutPreprocessing, DISABLED_SyncInferenceOneRemoteFrame) {
    // ---- Create workload context
    HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
    ASSERT_NE(nullptr, context.get());

    WorkloadID workloadId;
    context->setContext(workloadId);
    EXPECT_EQ(workloadId, context->getWorkloadContextID());
    EXPECT_EQ(HddlStatusCode::HDDL_OK, registerWorkloadContext(context));

    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load binary input
    const auto& inputTensor = PrecompiledResNet_Helper::resnet50_dpu_tensors.inputTensor;
    const auto& inputPath = PrecompiledResNet_Helper::resnet50_dpu.inputPath;
    auto inputRefBlob = make_blob_with_precision(inputTensor);
    inputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputPath, inputRefBlob));

    // ----- Allocate memory with HddlUnite on device
    RemoteMemoryFD remoteMemoryFd =
        allocateRemoteMemory(context, inputRefBlob->buffer().as<void*>(), inputRefBlob->size());

    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Import network providing context as input to bind to context
    const std::string& modelPath = PrecompiledResNet_Helper::resnet50_dpu.graphPath;

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
    const auto& outputPath = PrecompiledResNet_Helper::resnet50_dpu.outputPath;
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputPath, outputRefBlob));

    // --- Compare with expected output
    ASSERT_TRUE(outputBlob->byteSize() == outputRefBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == IE::Precision::U8);
    ASSERT_NO_THROW(Comparators::compareTopClasses(outputBlob, outputRefBlob, numberOfTopClassesToCompare));
}
