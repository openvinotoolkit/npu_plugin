// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_tests/set_blob.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common/functions.h"

using namespace LayerTestsUtils;

namespace BehaviorTestsDefinitions {

using VpuxBehaviorTestsSetBlob = SetBlobTest;

TEST_P(VpuxBehaviorTestsSetBlob, InternalPluginPrecisionConvert) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto backendName = getBackendName(*getCore());
    if (backendName.empty()) {
        GTEST_SKIP() << "No devices available. Test is skipped";
    } else if (backendName == "LEVEL0") {
        GTEST_SKIP() << "CumSum layer is not supported by MTL platform";
    }
#if defined(__arm__) || defined(__aarch64__)
    GTEST_SKIP() << "CumSum layer is not supported by ARM platform";
#endif
    SetBlobTest::Run();
}

}// namespace BehaviorTestsDefinitions


using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<Precision> precisionSet = {Precision::FP32, Precision::I16, Precision::U8, Precision::I8, Precision::U16, Precision::I32, Precision::BOOL,
                                             Precision::I64, Precision::U64};

const std::vector<setType> typeSet = {setType::INPUT, setType::OUTPUT, setType::BOTH};

const auto params = ::testing::Combine(::testing::ValuesIn(precisionSet),
                                       ::testing::ValuesIn(precisionSet),
                                       ::testing::ValuesIn(typeSet),
                                       ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY));

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTestsSetBlob, VpuxBehaviorTestsSetBlob, params, SetBlobTest::getTestCaseName);
