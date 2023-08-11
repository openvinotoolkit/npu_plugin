//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_KEEMBAY}}};

const std::vector<ov::AnyMap> autoConfigs = {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_KEEMBAY}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

const std::vector<ov::element::Type> prcs = {
        ov::element::boolean, ov::element::bf16, ov::element::f16, ov::element::f32, ov::element::f64, ov::element::i4,
        ov::element::i8,      ov::element::i16,  ov::element::i32, ov::element::i64, ov::element::u1,  ov::element::u4,
        ov::element::u8,      ov::element::u16,  ov::element::u32, ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Mutli_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

}  // namespace
