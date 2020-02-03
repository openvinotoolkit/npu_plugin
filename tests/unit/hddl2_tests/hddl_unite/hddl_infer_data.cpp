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

#include "hddl_unite/hddl2_infer_data.h"

using namespace vpu::HDDL2Plugin;
//------------------------------------------------------------------------------
//      class InferData_UnitTests Declaration
//------------------------------------------------------------------------------
class InferData_UnitTests : public ::testing::Test {};

TEST_F(InferData_UnitTests, constructor_default_NoThrow) { ASSERT_NO_THROW(HddlUniteInferData inferData); }

TEST_F(InferData_UnitTests, constructor_withNullContext_NoThrow) {
    auto context = nullptr;
    ASSERT_NO_THROW(HddlUniteInferData inferData(context));
}

TEST_F(InferData_UnitTests, prepareInput_nullBlob_Throw) {
    HddlUniteInferData inferData;
    const std::string inputName = "input";

    ASSERT_ANY_THROW(inferData.prepareInput(inputName, nullptr));
}

TEST_F(InferData_UnitTests, prepareOutput_nullBlob_Throw) {
    HddlUniteInferData inferData;
    const std::string outputName = "input";

    ASSERT_ANY_THROW(inferData.prepareOutput(outputName, nullptr));
}
