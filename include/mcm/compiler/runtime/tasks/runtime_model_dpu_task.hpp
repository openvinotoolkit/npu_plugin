#ifndef MV_RUNTIME_MODEL_DPU_TASK_
#define MV_RUNTIME_MODEL_DPU_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor.hpp"

namespace mv
{
    enum RuntimeModelDPULayerType
    {
        CONV,
        DWCONV,
        MAXPOOL,
        AVEPOOL,
        FCL,
        ELTWISE
    };

    enum RuntimeModelPPELayerType
    {
        STORE,
        LOAD,
        CLEAR,
        NOOP,
        HALT,
        ADD,
        SUB,
        MULT,
        LRELU,
        LRELUX,
        LRPRELU,
        MAXIMUM,
        MINIMUM,
        CEIL,
        FLOOR,
        AND,
        OR,
        XOR,
        NOT,
        ABS,
        NEG,
        POW,
        EXP,
        SIGMOID,
        TANH,
        SQRT,
        RSQRT,
        FLEXARB
    };

    enum RuntimeModelMPEMode
    {
        VECTOR,
        MATRIX
    };

    struct RuntimeModelPPEGenericTask
    {
        std::vector<RuntimeModelPPELayerType> ops_;
        unsigned clampValue_;
        unsigned ReLuNegSlope_;
        unsigned ReLuPosSlope_;
        unsigned pReLuAlpha_;
        RuntimeModelTensorReference scaleData_;
    };

    struct RuntimeModelDPUInvariantFields
    {
        RuntimeModelDPULayerType op_;
        RuntimeModelPPEGenericTask ppeTask_;
        unsigned clusterID_;

        unsigned kernelH_;
        unsigned kernelW_;
        unsigned kernelStrideH_;
        unsigned kernelStrideW_;

        unsigned padLeft_;
        unsigned padRight_;
        unsigned padTop_;
        unsigned padBottom_;

        RuntimeModelTensorReference inputData_;
        RuntimeModelTensorReference outputData_;
        RuntimeModelTensorReference weightsData_;
        RuntimeModelTensorReference biasData_;
        RuntimeModelMPEMode mpeMode_;
    };

    struct RuntimeModelDPUVariantFields
    {
        unsigned workloadID_;
        unsigned outputXIndex_;
        unsigned outputYIndex_;
        unsigned outputZIndex_;
    };

    struct RuntimeModelDPUTask : public RuntimeModelSpecificTask
    {
        RuntimeModelDPUInvariantFields invariant_;
        std::vector<RuntimeModelDPUVariantFields> variant_;
    };
}

#endif
