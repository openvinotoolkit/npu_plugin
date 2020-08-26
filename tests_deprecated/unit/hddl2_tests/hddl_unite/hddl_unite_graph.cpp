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
#include <vpux_compiler.hpp>
#include <network_desc.h>
#include <helper_remote_context.h>
#include "hddl2_unite_graph.h"
#include "hddl2_helpers/skip_conditions.h"

using namespace vpu::HDDL2Plugin;

//------------------------------------------------------------------------------
class HddlUniteGraph_UnitTests : public ::testing::Test {
// FIXME: Please take a note that _networkDescription should be destructed before _compiler,
// due _compiler is opened as plugin and _networkDescription is created by _compiler
// Need to design more accurate solution to avoid missunderstanding in future
// [Track number: S#37571]
protected:
    vpux::NetworkDescription_Helper _networkDescriptionHelper;

public:
    void SetUp() override;
    vpux::NetworkDescription::Ptr networkDescPtr = nullptr;
};

void HddlUniteGraph_UnitTests::SetUp() { networkDescPtr = _networkDescriptionHelper.getNetworkDesc(); }


//------------------------------------------------------------------------------
using HddlUniteGraph_Construct = HddlUniteGraph_UnitTests;
TEST_F(HddlUniteGraph_Construct, onlyGraph_NoThrow) {
    SKIP_IF_NO_DEVICE();
    ASSERT_NO_THROW(HddlUniteGraph hddlUniteGraph(networkDescPtr));
}

TEST_F(HddlUniteGraph_Construct, withContext_NoThrow) {
    SKIP_IF_NO_DEVICE();
    RemoteContext_Helper contextHelper;
    auto context = contextHelper.remoteContextPtr;

    ASSERT_NO_THROW(HddlUniteGraph hddlUniteGraph(networkDescPtr, context));
}

using HddlUniteGraph_Monkey = HddlUniteGraph_UnitTests;
TEST_F(HddlUniteGraph_Monkey, network_nullData_Throw) {
    SKIP_IF_NO_DEVICE();
    auto nullptrNetwork = nullptr;
    ASSERT_ANY_THROW(HddlUniteGraph hddlUniteGraph(nullptrNetwork));
}

TEST_F(HddlUniteGraph_Monkey, InferAsync_nullData_Throw) {
    SKIP_IF_NO_DEVICE();
    HddlUniteGraph hddlUniteGraph(networkDescPtr);
    ASSERT_ANY_THROW(hddlUniteGraph.InferAsync(nullptr));
}

