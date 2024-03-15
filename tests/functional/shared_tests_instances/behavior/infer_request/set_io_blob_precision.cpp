// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/infer_request/set_io_blob_precision.hpp"
#include "common/functions.h"
#include "common/utils.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<Precision> precisionSet = {Precision::FP32, Precision::I16, Precision::U8,
                                             Precision::I8,   Precision::U16, Precision::I32,
                                             Precision::BOOL, Precision::I64, Precision::U64};

const std::vector<setType> typeSet = {setType::INPUT, setType::OUTPUT, setType::BOTH};

const auto params = ::testing::Combine(::testing::ValuesIn(precisionSet), ::testing::ValuesIn(precisionSet),
                                       ::testing::ValuesIn(typeSet), ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_CASE_P(smoke_SetBlob, SetBlobTest, params, SetBlobTest::getTestCaseName);
