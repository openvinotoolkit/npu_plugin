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

#include <gtest/gtest.h>
#include <hddl2_helpers/helper_tensor_description.h>

#include "hddl_unite/hddl2_infer_data.h"

using namespace vpu::HDDL2Plugin;
class InferData_UnitTests : public ::testing::Test {
public:
    const bool enablePreProcessing = false;

    InferenceEngine::InputInfo::Ptr inputInfo;
    InferenceEngine::DataPtr outputInfo;

    InferenceEngine::Blob::Ptr blob;

protected:
    void SetUp() override;
    TensorDescription_Helper tensorDescriptionHelper;
};

void InferData_UnitTests::SetUp() {
    inputInfo = std::make_shared<InferenceEngine::InputInfo>();
    InferenceEngine::DataPtr inputData =
        std::make_shared<InferenceEngine::Data>("input", tensorDescriptionHelper.tensorDesc);
    inputInfo->setInputData(inputData);

    outputInfo = std::make_shared<InferenceEngine::Data>("output", tensorDescriptionHelper.tensorDesc);
    blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDescriptionHelper.tensorDesc);
}

//------------------------------------------------------------------------------
using InferData_constructor = InferData_UnitTests;
TEST_F(InferData_constructor, default_NoThrow) { ASSERT_NO_THROW(HddlUniteInferData inferData); }

TEST_F(InferData_constructor, withNullContext_NoThrow) {
    auto context = nullptr;
    ASSERT_NO_THROW(HddlUniteInferData inferData(enablePreProcessing, context));
}

//------------------------------------------------------------------------------
using InferData_prepareUniteInput = InferData_UnitTests;
TEST_F(InferData_prepareUniteInput, monkey_nullBlob_Throw) {
    HddlUniteInferData inferData;

    ASSERT_ANY_THROW(inferData.prepareUniteInput(nullptr, inputInfo));
}

TEST_F(InferData_prepareUniteInput, monkey_nullInfo_Throw) {
    HddlUniteInferData inferData;

    ASSERT_ANY_THROW(inferData.prepareUniteInput(blob, nullptr));
}

TEST_F(InferData_prepareUniteInput, monkey_nullDesc_Throw) {
    HddlUniteInferData inferData;
    inputInfo->setInputData(nullptr);

    ASSERT_ANY_THROW(inferData.prepareUniteInput(blob, inputInfo));
}

//------------------------------------------------------------------------------
using InferData_prepareUniteOutput = InferData_UnitTests;
TEST_F(InferData_prepareUniteOutput, monkey_nullBlob_Throw) {
    HddlUniteInferData inferData;

    ASSERT_ANY_THROW(inferData.prepareUniteOutput(nullptr, outputInfo));
}

TEST_F(InferData_prepareUniteOutput, monkey_nullDesc_Throw) {
    HddlUniteInferData inferData;
    InferenceEngine::Blob::Ptr blob;

    ASSERT_ANY_THROW(inferData.prepareUniteOutput(blob, nullptr));
}
