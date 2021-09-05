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
#ifdef CONFIG_TARGET_SOC_3720
    bool enqueTask(Op * operation,
                   const std::vector<Buffer> &inputs,
                   const std::vector<Buffer> &outputs,
                   int numSHAVEs,
                   PerformanceData *perfData);
#endif
    bool enqueTask(std::unique_ptr<MVCNN::UPALayerTaskT> && task,
                   const std::vector<Buffer> &inputs,
                   const std::vector<Buffer> &outputs,
                   int numSHAVEs,
                   PerformanceData *perfData);
    bool dequeResult();
};
