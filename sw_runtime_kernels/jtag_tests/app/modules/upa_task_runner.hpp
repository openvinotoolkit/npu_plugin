// {% copyright %}

#pragma once

#include <memory>
#include "mvTensor.h"
#include <Op.h>

#include <nn_perf_measurement.h>

/**
 * @brief wrapper over UPA runner, svu runtime, etc
 */
class UPATaskRunner {
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
