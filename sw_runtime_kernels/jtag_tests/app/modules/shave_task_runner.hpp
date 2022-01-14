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

#pragma once

#include <memory>
#include "mvTensor.h"
#include <Op.h>

#include <nn_perf_measurement.h>

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
};
