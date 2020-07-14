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

#include "hddl2_graph.h"

#include "gtest/gtest.h"
#include "hddl2_plugin.h"
#include "mcm_config.h"
#include "models/model_pooling.h"
#include "models/precompiled_resnet.h"

using namespace vpu::HDDL2Plugin;
using namespace InferenceEngine;

//------------------------------------------------------------------------------
enum typeOfGraph { fromImportedGraph, fromCompiledGraph };

//------------------------------------------------------------------------------
class Graph_Common_UnitTests : public ::testing::Test, public ::testing::WithParamInterface<typeOfGraph> {
public:
    void SetUp() override;

    Graph::Ptr graphPtr;

    struct PrintToStringParamName {
        std::string operator()(testing::TestParamInfo<typeOfGraph> const& info) const;
    };

private:
    const vpu::MCMConfig _defaultMCMConfig;
};

void Graph_Common_UnitTests::SetUp() {
    if (GetParam() == fromImportedGraph) {
        const std::string _modelToImport = PrecompiledResNet_Helper::resnet50.graphPath;
        ASSERT_NO_THROW(graphPtr = std::make_shared<ImportedGraph>(_modelToImport, _defaultMCMConfig));
    } else {
        ModelPooling_Helper modelPoolingHelper;
        CNNNetwork network = modelPoolingHelper.getNetwork();
        ASSERT_NO_THROW(graphPtr = std::make_shared<CompiledGraph>(network, _defaultMCMConfig));
    }
}

std::string Graph_Common_UnitTests::PrintToStringParamName::operator()(
    const testing::TestParamInfo<typeOfGraph>& info) const {
    auto createdFrom = info.param;
    if (createdFrom == fromImportedGraph) {
        return "fromImportedGraph";
    } else if (createdFrom == fromCompiledGraph) {
        return "fromCompiledGraph";
    } else {
        return "Unknown params";
    }
}


//------------------------------------------------------------------------------
TEST_P(Graph_Common_UnitTests, getDeviceName_ReturnNotNull) {
    const std::string name = graphPtr->getGraphName();
    ASSERT_GT(name.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getInputsInfo_ReturnNotEmpty) {
    auto inputsInfo = graphPtr->getInputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getOutputsInfo_ReturnNotEmpty) {
    auto inputsInfo = graphPtr->getOutputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

TEST_P(Graph_Common_UnitTests, getGraphBlob_ReturnNotEmpty) {
    auto graphBlob = graphPtr->getGraphBlob();
    ASSERT_GT(graphBlob.size(), 0);
}

//------------------------------------------------------------------------------
const static std::vector<typeOfGraph> createdFrom = {fromImportedGraph, fromCompiledGraph};

INSTANTIATE_TEST_CASE_P(GraphFrom, Graph_Common_UnitTests, ::testing::ValuesIn(createdFrom),
    Graph_Common_UnitTests::PrintToStringParamName());
