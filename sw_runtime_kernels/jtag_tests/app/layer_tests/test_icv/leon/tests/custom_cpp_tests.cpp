//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "custom_cpp_test_base.h"
#include "custom_cpp_tests.h"
#include <sw_nn_runtime_types.h>

uint64_t cmxParamContainer[((nn::shave_lib::MAX_INPUT_TENSORS + nn::shave_lib::MAX_OUTPUT_TENSORS) * sizeof(OpTensor) + 1024) / sizeof(uint64_t)] __attribute__((section(".nncmx0.shared.data")));
