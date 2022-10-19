//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <memory>
#include "mvTensor.h"
#include <Op.h>

#include <nn_perf_measurement.h>
#include <nn_shave_manager.h>

/**
 * @brief wrapper over UPA runner, svu runtime, etc
 */
class ShaveTaskRunner {
    bool _enqued = false;
 public:
    /**
     * @breif enque certain task with giver inputs and outputs buffers
     * @return true if succeeded
     */
    bool enqueTask(Op * operation,
                   const std::vector<OpTensor> &inputs,
                   const std::vector<OpTensor> &outputs,
                   int numSHAVEs,
                   PerformanceData *perfData);
    bool dequeResult();
    std::shared_ptr<nn::inference_runtime::shaves::ShaveManager> _shaveManager;
};
