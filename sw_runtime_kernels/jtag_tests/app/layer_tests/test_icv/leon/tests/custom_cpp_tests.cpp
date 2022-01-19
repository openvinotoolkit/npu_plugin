//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "custom_cpp_test_base.h"
#include "custom_cpp_tests.h"
#include <sw_nn_runtime_types.h>

uint64_t cmxParamContainer[((nn::shave_lib::MAX_INPUT_TENSORS + nn::shave_lib::MAX_OUTPUT_TENSORS) * sizeof(OpTensor) + 1024) / sizeof(uint64_t)] __attribute__((section(".nncmx0.shared.data")));
