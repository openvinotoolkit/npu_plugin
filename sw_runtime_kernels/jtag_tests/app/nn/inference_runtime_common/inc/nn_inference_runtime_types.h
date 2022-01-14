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

#ifndef NN_INFERENCE_RUNTIME_TYPES_H_
#define NN_INFERENCE_RUNTIME_TYPES_H_

// TODO: This config split can be more tightly consolidated into one header.
#if (defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_3100))
#include <sw_nn_runtime_types.h>
#include <nn_runtime_types.h>
#include <nn_relocation.h>
#include <nn_memory.h>
#include <nn_memory_alloc.h>
#include <array>
#include <algorithm>
#include <nn_perf_measurement.h>
#   include "nn_inference_runtime_types_2490.h"
#endif // CONFIG_TARGET_SOC_MA2490 || CONFIG_TARGET_SOC_3100

#endif // NN_INFERENCE_RUNTIME_TYPES_H_
