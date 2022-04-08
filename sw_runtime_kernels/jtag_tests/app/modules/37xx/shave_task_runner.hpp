//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <memory>
#include "mvTensor.h"
#include <CustomCpp.h>

// #include <nn_perf_measurement.h>
#include <nn_shave_manager.h>
#include <nn_cmx_memory_map.h>

using namespace nn;

const uint32_t NN_CMX_BASE = 0x2e000000;
#define SHAVE_STACK_SIZE 0x400

class ShaveTaskRunner {
 public:

    ShaveTaskRunner();
    ShaveTaskRunner(const ShaveTaskRunner &) = delete;
    ShaveTaskRunner &operator=(const ShaveTaskRunner &) = delete;
    ~ShaveTaskRunner();

    bool enqueTask(CustomCppLayerParams ops);//,
                   //PerformanceData *perfData);
    bool dequeResult();

private:
    bool _enqued = false;
    common_runtime::NNCmxMemoryMap *nnCmx_;
    common_runtime::StaticMapping globalAreas_ NN_CACHE_ALIGNED;
    inference_runtime::shaves::ShaveManager shave_manager_;
    common_runtime::NNShaveRuntimeConfigs actRtConfigs;
    act_runtime::ActKernelRange kRange;
    act_runtime::ActKernelInvocation kInvo;
    common_runtime::ParsedShaveElfs shaveElfs;
};
