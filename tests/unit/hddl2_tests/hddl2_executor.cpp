//
// Copyright 2020 Intel Corporation.
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

#include <gtest/gtest.h>
#include "hddl2_executor.h"
#include "helpers/network_desc.h"
#include "hddl2_helpers/skip_conditions.h"

using namespace vpux::hddl2;

class Executor_UnitTests: public ::testing::Test {
public:
    vpux::NetworkDescription::Ptr networkDescPtr = nullptr;
    const vpux::VPUXConfig config;
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
    auto executor = HDDL2Executor::prepareExecutor(networkDescPtr, config, nullptr);
    ASSERT_EQ(executor, nullptr);
}

//------------------------------------------------------------------------------
using Executor_WithDevice = Executor_UnitTests;
TEST_F(Executor_WithDevice, createExecutor_WithDevice_ReturnNotNull) {
    SKIP_IF_NO_DEVICE();
    auto executor = HDDL2Executor::prepareExecutor(networkDescPtr, config, nullptr);
    ASSERT_NE(executor, nullptr);
}
