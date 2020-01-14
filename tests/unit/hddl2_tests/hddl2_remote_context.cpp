//
// Copyright 2019 Intel Corporation.
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

#include "helpers/hddl2_remote_context.h"

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include "hddl2_helpers/hddl2_workload_context.h"
#include "hddl2_params.hpp"
#include "hddl2_plugin.h"
#include "helpers/hddl2_device_emulator.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Declaration
//------------------------------------------------------------------------------
class HDDL2_RemoteContext_UnitTests : public ::testing::Test {
public:
    void SetUp() override;

    InferenceEngine::ParamMap params;
    HDDL2_WorkloadContext_Helper workloadContextHelper;
};

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Implementation
//------------------------------------------------------------------------------
void HDDL2_RemoteContext_UnitTests::SetUp() {
    WorkloadID id = workloadContextHelper.getWorkloadId();
    params = HDDL2_With_RemoteContext_Helper::wrapWorkloadIdToMap(id);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Initiations - constructor
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromCorrectId_NoThrow) {
    ASSERT_NO_THROW(HDDL2RemoteContext context(params));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_TwiceSameId_NoThrow) {
    HDDL2RemoteContext::Ptr firstContext = nullptr;
    HDDL2RemoteContext::Ptr secondContext = nullptr;

    ASSERT_NO_THROW(firstContext = std::make_shared<HDDL2RemoteContext>(params));
    ASSERT_NO_THROW(secondContext = std::make_shared<HDDL2RemoteContext>(params));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_TwiceSameId_NoNull) {
    HDDL2RemoteContext::Ptr firstContext = nullptr;
    HDDL2RemoteContext::Ptr secondContext = nullptr;

    firstContext = std::make_shared<HDDL2RemoteContext>(params);
    secondContext = std::make_shared<HDDL2RemoteContext>(params);

    ASSERT_NE(nullptr, firstContext);
    ASSERT_NE(nullptr, secondContext);
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromEmptyPararams_ThrowException) {
    InferenceEngine::ParamMap emptyParams = {};

    ASSERT_ANY_THROW(HDDL2RemoteContext context(emptyParams));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromIncorrectPararams_ThrowException) {
    InferenceEngine::ParamMap badParams = {{"Bad key", "Bad value"}};
    ASSERT_ANY_THROW(HDDL2RemoteContext context(badParams));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromNotExistsContext_ThrowException) {
    const int badWorkloadId = UINT32_MAX;
    EXPECT_FALSE(workloadContextHelper.isValidWorkloadContext(badWorkloadId));

    InferenceEngine::ParamMap notExistsParams = HDDL2_With_RemoteContext_Helper::wrapWorkloadIdToMap(badWorkloadId);

    ASSERT_ANY_THROW(HDDL2RemoteContext context(notExistsParams));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Initiations - destructor
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, destructor_workloadContextNotUnregistered) {
    int workloadId = workloadContextHelper.getWorkloadId();
    auto params = HDDL2_With_RemoteContext_Helper::wrapWorkloadIdToMap(workloadId);

    { HDDL2RemoteContext context(params); }
    ASSERT_TRUE(workloadContextHelper.isValidWorkloadContext(workloadId));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Initiations - getDeviceName
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, getDeviceName_ReturnEmulatorName) {
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params);

    ASSERT_EQ(emulatorDeviceNameInPlugin, context->getDeviceName());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Initiations - getParams
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, getParams_containsWorkloadId) {
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params);

    const std::string workloadContextKey = IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID);

    auto params = context->getParams();
    auto iter = params.find(workloadContextKey);

    ASSERT_TRUE(iter != params.end());
}

TEST_F(HDDL2_RemoteContext_UnitTests, getParams_containsSameWorkloadId) {
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params);
    const std::string workloadContextKey = IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID);

    auto params = context->getParams();
    auto iter = params.find(workloadContextKey);
    const uint64_t workloadId = iter->second.as<uint64_t>();

    const uint64_t correctWorkloadId = workloadContextHelper.getWorkloadId();
    ASSERT_EQ(correctWorkloadId, workloadId);
}
