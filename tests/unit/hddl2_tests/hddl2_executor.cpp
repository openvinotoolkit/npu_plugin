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

#include "hddl2_executor.h"
#include <gtest/gtest.h>
#include "hddl2_helpers/skip_conditions.h"
#include "helpers/network_desc.h"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/mcm_compiler.hpp"
#include "vpux/al/config/runtime.hpp"

using namespace vpux::hddl2;

class Executor_UnitTests : public ::testing::Test {
public:
    vpux::NetworkDescription::Ptr networkDescPtr = nullptr;

    std::shared_ptr<vpux::OptionsDesc> options;
    vpux::Config config;

    Executor_UnitTests(): options(std::make_shared<vpux::OptionsDesc>()), config(options) {
    }

protected:
    void SetUp() override;
};

void Executor_UnitTests::SetUp() {
    vpux::registerCommonOptions(*options);
    vpux::registerCompilerOptions(*options);
    vpux::registerMcmCompilerOptions(*options);
    vpux::registerRunTimeOptions(*options);

    if (isEmulatorDevice())
        return;

    vpux::NetworkDescription_Helper networkDescriptionHelper;
    networkDescPtr = networkDescriptionHelper.getNetworkDesc();
}

//------------------------------------------------------------------------------
using Executor_NoDevice = Executor_UnitTests;
TEST_F(Executor_NoDevice, createExecutor_NoDevice_ReturnNull) {
    SKIP_IF_DEVICE();
    if (isEmulatorDevice())
        GTEST_SKIP() << "Test not intended for emulator run.";

    auto executor = HDDL2Executor::prepareExecutor(networkDescPtr, config, nullptr);
    ASSERT_EQ(executor, nullptr);
}

//------------------------------------------------------------------------------
using Executor_WithDevice = Executor_UnitTests;
TEST_F(Executor_WithDevice, createExecutor_WithDevice_ReturnNotNull) {
    SKIP_IF_NO_DEVICE();
    if (isEmulatorDevice())
        GTEST_SKIP() << "Test not intended for emulator run.";

    auto executor = HDDL2Executor::prepareExecutor(networkDescPtr, config, nullptr);
    ASSERT_NE(executor, nullptr);
}
