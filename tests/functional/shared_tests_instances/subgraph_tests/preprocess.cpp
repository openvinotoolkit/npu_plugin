// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"


using namespace SubgraphTestsDefinitions;

// [Track number: S#69189]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_PrePostProcess, PrePostProcessTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(ov::builder::preprocess::generic_preprocess_functions()),
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                         PrePostProcessTest::getTestCaseName);
