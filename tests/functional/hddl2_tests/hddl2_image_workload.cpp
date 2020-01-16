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

#include <ie_core.hpp>

#include "file_reader.h"
#include "gtest/gtest.h"
#include "parametric_executable_network.h"
#include "ie_blob.h"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_ImageWorkload_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_ImageWorkload_Tests : public Executable_Network_Parametric {
public:
    modelBlobInfo blobInfo = PrecompiledResNet_Helper::resnet;
};

//------------------------------------------------------------------------------
//      class HDDL2_ImageWorkload_Tests Initiation
//------------------------------------------------------------------------------
/**
 * 1. Create executable network
 * 2. Set input data
 * 3. Run inference
 * 4. Check output
 */

TEST_P(HDDL2_ImageWorkload_Tests, SyncInference) {
    // FIXME TODO Fix load network case
    if (GetParam() == LoadNetwork) {
        SKIP();
    }

    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Import or load network. Already prepared for test
    (void)executableNetwork;

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    IE::ConstInputsDataMap inputInfo;
    inputInfo = executableNetwork.GetInputsInfo();
    std::string inputFilePath = blobInfo.inputPath;

    for (auto& item : inputInfo) {
        IE::Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
        auto size = inputBlob->size();
        printf("=== Size %lu\n", size);
        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
    }

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // FIXME Unfinished
    // TODO How to check that data is correct ?
    IE::Blob::Ptr outputBlob = inferRequest.GetBlob("output");
    auto data = outputBlob->buffer().as<uint8_t*>();
    (void)data;
}

//------------------------------------------------------------------------------
//      class HDDL2_ImageWorkload_Tests Test case Initiations
//------------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(ExecNetworkFrom, HDDL2_ImageWorkload_Tests, ::testing::ValuesIn(memoryOwners),
    Executable_Network_Parametric::PrintToStringParamName());
