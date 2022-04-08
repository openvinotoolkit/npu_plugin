// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/ov_executable_network/exec_graph_info.hpp"

#include <common_test_utils/test_constants.hpp>

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type_t> netPrecisions = {
        ov::element::f16,
        ov::element::f32,
};
const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_BehaviorTests, OVExecGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         OVExecGraphImportExportTest::getTestCaseName);

}  // namespace
