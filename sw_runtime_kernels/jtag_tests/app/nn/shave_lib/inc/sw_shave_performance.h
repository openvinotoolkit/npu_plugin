/*
* {% copyright %}
*/
#pragma once

#include "sw_layer_params.h"
#include "sw_tensor_ref.h"

namespace nn {
namespace shave_lib {

typedef struct {
    uint64_t instrs;
    uint64_t cycles;
    uint64_t stalls;
    uint64_t branches;
} MvPerfStruct;

class PerformanceCounters {
public:
    PerformanceCounters(MvPerfStruct *perf, uint32_t shaveID);
    ~PerformanceCounters();

    void measureBegin();
    void measureEnd();

private:
    uint32_t svuBase{};
    MvPerfStruct *layerParamPerf{};
    MvPerfStruct currentPerf{};
};

} // namespace shave_lib
} // namespace nn
