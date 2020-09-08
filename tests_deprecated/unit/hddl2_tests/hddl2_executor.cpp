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
#include "hddl2_executor.h"
#include "helpers/network_desc.h"
#include "hddl2_helpers/skip_conditions.h"

using namespace vpux::HDDL2;

class Executor_UnitTests: public ::testing::Test {
public:
    vpux::NetworkDescription::Ptr networkDescPtr = nullptr;

protected:
    void SetUp() override;
};

void Executor_UnitTests::SetUp() {
    vpux::NetworkDescription_Helper networkDescriptionHelper;
    networkDescPtr = networkDescriptionHelper.getNetworkDesc();
}

//------------------------------------------------------------------------------
using Executor_NoDevice = Executor_UnitTests;
TEST_F(Executor_NoDevice, createExecutor_NoDevice_ReturnNull) {
    SKIP_IF_DEVICE();
    auto executor = HDDL2Executor::prepareExecutor(networkDescPtr);
    ASSERT_EQ(executor, nullptr);
}

//------------------------------------------------------------------------------
using Executor_WithDevice = Executor_UnitTests;
TEST_F(Executor_WithDevice, createExecutor_WithDevice_ReturnNotNull) {
    SKIP_IF_NO_DEVICE();
    auto executor = HDDL2Executor::prepareExecutor(networkDescPtr);
    ASSERT_NE(executor, nullptr);
}
