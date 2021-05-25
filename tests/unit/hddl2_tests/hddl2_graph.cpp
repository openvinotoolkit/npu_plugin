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

#include "gtest/gtest.h"
#include "models/model_pooling.h"
#include "models/precompiled_resnet.h"
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
};

void Graph_Common_UnitTests::SetUp() {
    auto compiler = vpux::Compiler::create();
    if (GetParam() == fromImportedGraph) {
        const std::string modelToImport = PrecompiledResNet_Helper::resnet50.graphPath;
        ASSERT_NO_THROW(networkPtr = compiler->parse(modelToImport));
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
    const std::string name = networkPtr->getName();
    ASSERT_GT(name.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getInputsInfo_ReturnNotEmpty) {
    auto inputsInfo = networkPtr->getInputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getOutputsInfo_ReturnNotEmpty) {
    auto inputsInfo = networkPtr->getOutputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getGraphBlob_ReturnNotEmpty) {
    auto graphBlob = networkPtr->getCompiledNetwork();
    ASSERT_GT(graphBlob.size(), 0);
}

//------------------------------------------------------------------------------
const static std::vector<typeOfGraph> createdFrom = {fromImportedGraph};

INSTANTIATE_TEST_CASE_P(GraphFrom, Graph_Common_UnitTests, ::testing::ValuesIn(createdFrom),
    Graph_Common_UnitTests::PrintToStringParamName());
