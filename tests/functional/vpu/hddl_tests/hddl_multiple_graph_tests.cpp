// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include <gtest/gtest.h>

#include "myriad_layers_tests.hpp"
#include "tests_vpu_common.hpp"
#include "tests_hddl_utils.hpp"

using namespace InferenceEngine;

class myriadLayersNetworkLoadingTestsHDDL_nightly : public myriadLayersTests_nightly {
    void SetUp() override {
        if (!hasHDDL_R()) {
            return;
        }
#if defined(_WIN32) || defined(WIN32)
        std::cout << "Disable for windows till hddldaemon cannot be run from the test, CVS-14658";
        return;
#endif
        myriadLayersTests_nightly::SetUp();
        HDDLTestsUtils::killHddlService();

        ASSERT_TRUE(HDDLTestsUtils::runHddlService("hddl_service_bypass_device_number_1.config"));
    }

    void TearDown() override {
        myriadLayersTests_nightly::TearDown();

        ASSERT_TRUE(HDDLTestsUtils::killHddlService());
    }
};

// CVS-16096
TEST_F(myriadLayersNetworkLoadingTestsHDDL_nightly, CanLoadNetworkAfterUnsuccessfulLoading) {
    if (!hasHDDL_R()) {
        SKIP() << "Skip for non hddl plugin";
    }

#if defined(_WIN32) || defined(WIN32)
    SKIP() << "Disable for windows till hddldaemon cannot be run from the test, CVS-14658";
#endif


    const std::size_t _200MB = 200;
    const std::size_t _1MB = 1;
    // create several networks to avoid cashing the network by hddl-service
    auto enormousNetwork0 = createNetworkWithDesiredSize(_200MB);
    auto enormousNetwork1 = createNetworkWithDesiredSize(_200MB);
    auto enormousNetwork2 = createNetworkWithDesiredSize(_200MB);
    auto tinyNetwork = createNetworkWithDesiredSize(_1MB);

    std::vector<IExecutableNetwork::Ptr> exeNetwork(4);
    config_t networkConfig = {{VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "tag"}};
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(exeNetwork[0], *enormousNetwork0, networkConfig, &_resp)) << _resp.msg;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(exeNetwork[1], *enormousNetwork1, networkConfig, &_resp)) << _resp.msg;

    ASSERT_NE(StatusCode::OK, myriadPluginPtr->LoadNetwork(exeNetwork[2], *enormousNetwork2, networkConfig, &_resp)) << _resp.msg;

    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(exeNetwork[3], *tinyNetwork, networkConfig, &_resp)) << _resp.msg;
}
