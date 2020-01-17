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
#include "hddl2_helpers/helper_model_loader.h"
#include "hddl2_helpers/helper_precompiled_resnet.h"
#include "hddl2_plugin.h"

using namespace vpu::HDDL2Plugin;
using namespace InferenceEngine;

//------------------------------------------------------------------------------
//     class HDDL2_Graph_UnitTests Params
//------------------------------------------------------------------------------
enum typeOfGraph { ImportedGraph, CompiledGraph };

//------------------------------------------------------------------------------
//     class HDDL2_Graph_UnitTests Declaration
//------------------------------------------------------------------------------
class HDDL2_Graph_UnitTests : public ::testing::WithParamInterface<typeOfGraph>, public ::testing::Test {
public:
    void SetUp() override;

    HDDL2Graph::Ptr graphPtr;

    struct PrintToStringParamName {
        std::string operator()(testing::TestParamInfo<typeOfGraph> const& info) const;
    };

private:
    const std::string _modelToImport = PrecompiledResNet_Helper::resnet.graphPath;
    const std::string _modelToCompile = "googlenet/bvlc_googlenet_fp16";
};

void HDDL2_Graph_UnitTests::SetUp() {
    if (GetParam() == ImportedGraph) {
        ASSERT_NO_THROW(graphPtr = std::make_shared<HDDL2ImportedGraph>(_modelToImport));
    } else {
        CNNNetwork network;
        ModelLoader_Helper::LoadModel(_modelToCompile, network);
        ASSERT_NO_THROW(graphPtr = std::make_shared<HDDL2CompiledGraph>(network));
    }
}

std::string HDDL2_Graph_UnitTests::PrintToStringParamName::operator()(
    const testing::TestParamInfo<typeOfGraph>& info) const {
    auto createdFrom = info.param;
    if (createdFrom == ImportedGraph) {
        return "ImportedGraph";
    } else if (createdFrom == CompiledGraph) {
        return "CompiledGraph";
    } else {
        return "Unknown params";
    }
}

//------------------------------------------------------------------------------
//     HDDL2_Graph_UnitTests Initiation
//------------------------------------------------------------------------------
TEST_P(HDDL2_Graph_UnitTests, getDeviceName_ReturnNotNull) {
    const std::string name = graphPtr->getGraphName();
    ASSERT_GT(name.size(), 0);
}

TEST_P(HDDL2_Graph_UnitTests, getInputsInfo_ReturnNotEmpty) {
    auto inputsInfo = graphPtr->getInputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

TEST_P(HDDL2_Graph_UnitTests, getOutputsInfo_ReturnNotEmpty) {
    auto inputsInfo = graphPtr->getOutputsInfo();
    ASSERT_GT(inputsInfo.size(), 0);
}

//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Test case Initiations
//------------------------------------------------------------------------------
// TODO Enable CompiledGraph after implementing graph compilation (enable mcm compiler)
const static std::vector<typeOfGraph> createdFrom = {ImportedGraph};

INSTANTIATE_TEST_CASE_P(GraphFrom, HDDL2_Graph_UnitTests, ::testing::ValuesIn(createdFrom),
    HDDL2_Graph_UnitTests::PrintToStringParamName());

//------------------------------------------------------------------------------
//     class HDDL2_Graph_UnitTests Declaration
//------------------------------------------------------------------------------
class HDDL2_ImportedGraph_UnitTests : public ::testing::Test {
public:
    void SetUp() override;
    HDDL2Graph::Ptr graphPtr;

private:
    const std::string _modelToImport = PrecompiledResNet_Helper::resnet.graphPath;
};

void HDDL2_ImportedGraph_UnitTests::SetUp() {
    ASSERT_NO_THROW(graphPtr = std::make_shared<HDDL2ImportedGraph>(_modelToImport));
}

//------------------------------------------------------------------------------
//     HDDL2_ImportedGraph_UnitTests Initiation
//------------------------------------------------------------------------------
TEST_F(HDDL2_ImportedGraph_UnitTests, getGraphBlob_NotEmptyBlob) {
    auto graphContent = graphPtr->getGraphBlob();
    ASSERT_GT(graphContent.size(), 0);
}
