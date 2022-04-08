//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
