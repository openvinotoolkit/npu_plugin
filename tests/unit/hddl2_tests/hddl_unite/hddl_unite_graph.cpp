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

#include "hddl_unite/hddl2_unite_graph.h"
#include "helper_graph.h"
#include "helper_remote_context.h"

using namespace vpu::HDDL2Plugin;
//------------------------------------------------------------------------------
//      class HddlUniteGraph_UnitTests Declaration
//------------------------------------------------------------------------------
class HddlUniteGraph_UnitTests : public ::testing::Test {
public:
    void SetUp() override;
    HDDL2Graph::Ptr graph;

protected:
    ImportedGraph_Helper _importedGraphHelper;
};

void HddlUniteGraph_UnitTests::SetUp() { graph = _importedGraphHelper.getGraph(); }

//------------------------------------------------------------------------------
//      class HddlUniteGraph_UnitTests Initiation
//------------------------------------------------------------------------------
TEST_F(HddlUniteGraph_UnitTests, constructor_onlyGraph_NoThrow) {
    ASSERT_NO_THROW(HddlUniteGraph hddlUniteGraph(graph));
}

TEST_F(HddlUniteGraph_UnitTests, constructor_withContext_NoThrow) {
    RemoteContext_Helper contextHelper;
    auto context = contextHelper.remoteContextPtr;

    ASSERT_NO_THROW(HddlUniteGraph hddlUniteGraph(graph, context));
}
