// Copyright (C) 2019 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "subgraph_tests/split_conv_concat.hpp"

#include <vector>

#include "vpu_ov1_layer_test.hpp"

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> input_types = {ov::element::f32, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_NoReshape, SplitConvConcat,
                         ::testing::Combine(::testing::ValuesIn(input_types),
                                            ::testing::Values(ov::Shape{1, 6, 40, 40}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         SplitConvConcat::getTestCaseName);
}  // namespace
