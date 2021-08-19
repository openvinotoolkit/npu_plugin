// {% copyright %}

#pragma once

#include <memory>
#include "include/software_generated.h"
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
    bool enqueTask(std::unique_ptr<MVCNN::UPALayerTaskT> && task,
                   const std::vector<Buffer> &inputs,
                   const std::vector<Buffer> &outputs,
                   int numSHAVEs,
                   PerformanceData *perfData);
    bool dequeResult();
};
