//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "gtest/gtest.h"
#include "hddl2_helpers/skip_conditions.h"
#include "models/model_pooling.h"
#include "simple_graph.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/mcm_compiler.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux_plugin.h"

using namespace InferenceEngine;

//------------------------------------------------------------------------------
enum typeOfGraph { fromImportedGraph };

//------------------------------------------------------------------------------
class Graph_Common_UnitTests : public ::testing::Test, public ::testing::WithParamInterface<typeOfGraph> {
public:
    void SetUp() override;

    vpux::NetworkDescription::Ptr networkPtr = nullptr;
    struct PrintToStringParamName {
        std::string operator()(testing::TestParamInfo<typeOfGraph> const& info) const;
    };

protected:
    std::stringstream _blobStream;
};

void Graph_Common_UnitTests::SetUp() {
    if (isEmulatorDevice())
        return;

    auto options = std::make_shared<vpux::OptionsDesc>();
    vpux::registerCommonOptions(*options);
    vpux::registerCompilerOptions(*options);
    vpux::registerMcmCompilerOptions(*options);
    vpux::registerRunTimeOptions(*options);

    vpux::Config config(options);

    auto compiler = vpux::Compiler::create(config);
    utils::simpleGraph::getExeNetwork()->Export(_blobStream);
    if (GetParam() == fromImportedGraph) {
        ASSERT_NO_THROW(networkPtr = compiler->parse(_blobStream, config, ""));
    }
}

std::string Graph_Common_UnitTests::PrintToStringParamName::operator()(
        const testing::TestParamInfo<typeOfGraph>& info) const {
    auto createdFrom = info.param;
    if (createdFrom == fromImportedGraph) {
        return "fromImportedGraph";
    } else {
        return "Unknown params";
    }
}

//------------------------------------------------------------------------------
TEST_P(Graph_Common_UnitTests, getDeviceName_ReturnNotNull) {
    if (isEmulatorDevice())
        GTEST_SKIP() << "Test not intended for emulator run.";
    const std::string name = networkPtr->getName();
    ASSERT_GT(name.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getInputsInfo_ReturnNotEmpty) {
    if (isEmulatorDevice())
        GTEST_SKIP() << "Test not intended for emulator run.";
    auto inputsInfo = networkPtr->getInputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getOutputsInfo_ReturnNotEmpty) {
    if (isEmulatorDevice())
        GTEST_SKIP() << "Test not intended for emulator run.";
    auto inputsInfo = networkPtr->getOutputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getGraphBlob_ReturnNotEmpty) {
    if (isEmulatorDevice())
        GTEST_SKIP() << "Test not intended for emulator run.";
    auto graphBlob = networkPtr->getCompiledNetwork();
    ASSERT_GT(graphBlob.size(), 0);
}

//------------------------------------------------------------------------------
const static std::vector<typeOfGraph> createdFrom = {fromImportedGraph};

INSTANTIATE_TEST_SUITE_P(GraphFrom, Graph_Common_UnitTests, ::testing::ValuesIn(createdFrom),
                         Graph_Common_UnitTests::PrintToStringParamName());
