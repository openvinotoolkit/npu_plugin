#ifndef MV_RUNTIME_MODEL_DPU_TASK_
#define MV_RUNTIME_MODEL_DPU_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "include/mcm/compiler/runtime/tasks/runtime_model_nn_tensor_task.hpp"

namespace mv
{
    enum RuntimeModelDPULayerType
    {
        CONV,
        DWCONV,
        MAXPOOL,
        AVEPOOL,
        FCL,
        ELTWISE,
        IDENTITY
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

    struct RuntimeModelPPEFixedFunction
    {
        std::vector<RuntimeModelPPELayerType*> ops;
        unsigned clampValueLow;
        unsigned clampValueHigh;
        unsigned ReLuNegSlope;
        unsigned ReLuPosSlope;
        unsigned pReLuAlpha;
    };

    struct RuntimeModelPPEGenericTask
    {
        RuntimeModelTensorReference * scaleData;
        std::vector<RuntimeModelPPEFixedFunction*> fixedFunction;
    };

    struct RuntimeModelDPUInvariantFields
    {
        RuntimeModelDPULayerType * op;
        RuntimeModelPPEGenericTask * ppeTask;
        std::vector<RuntimeModelNNTask*> nnvShvTask;

        unsigned kernelH;
        unsigned kernelW;
        unsigned kernelStrideH;
        unsigned kernelStrideW;

        RuntimeModelTensorReference * inputData;
        RuntimeModelTensorReference * outputData;
        RuntimeModelTensorReference * weightsData;
        RuntimeModelTensorReference * biasData;
    };

    struct RuntimeModelDPUVariantFields
    {
        unsigned clusterID;
        unsigned workloadID;
        RuntimeModelMPEMode * mpeMode;

        unsigned padLeft;
        unsigned padRight;
        unsigned padTop;
        unsigned padBottom;

        unsigned workloadStartX;
        unsigned workloadStartY;
        unsigned workloadStartZ;
        unsigned workloadEndX;
        unsigned workloadEndY;
        unsigned workloadEndZ;
    };

    struct RuntimeModelDPUTask : public RuntimeModelSpecificTask
    {
      /// This object describes an entire network layer that will be operated
      /// on by the NCE's DPUs, PPEs and NNSHV Assist Library.
      ///
      /// The layer is likely to be split into different "workloads" -
      /// subsections of the layer for the processors to split work upon.
      ///
      /// Fields common to these subsections are to be stored in the 'invariant'
      /// part of this object.
      /// All per-section information should be contained in the 'variant'
      /// vector. Where there is one entry per 'workload'. If there are no unique
      /// information per-workload, empty objects should still be placed.
      /// There is a 1-to-1 Relationship between DPUs and "Workloads"
      ///
      /// Below is a typical example of splitting a Convolution
      //// across 5 DPUs (1 Cluster)
      ///                                XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
      ///                         XXXXXXX          XXXXX              XXXXXX
      ///                     XXXXX            XXXXX               XXX     X
      ///                XXXXX             XXXXX                XXXX       X
      ///         XXXXXXXX             XXXXX               XXXXX           X
      ///    XXXXXX                XXXXX             X XX X               XX
      /// XX-------------------+XXX----------------+XX                 XXXXX
      /// |                    |                   |               XXXX    X
      /// |                    |                   |           XXXXX       X
      /// |                    |        C          |      XXXXXX           X
      /// |         A          |                   |  XXXX               XXX
      /// |                    +-------------------+XXX               XXXX X
      /// |                    |                   |               XXXX    X
      /// |                    |                   |            XXXX       X
      /// +--------------------+        D          |        XXXXX          X
      /// |                    |                   |    XXXXX           X XX
      /// |                    |                   | XXX             XXXX
      /// |                    +-------------------XX             XXXX
      /// |         B          |                   |           XXXX
      /// |                    |                   |        XXXX
      /// |                    |        E          |      XXX
      /// |                    |                   |   XX
      /// +--------------------+-------------------+XX
      ///
      /// Splits for workloads are not limited to different dimensions when using a
      /// Single Cluster.
      /// However, splitting across clusters is limited to splitting over height
      /// and splitting over channels.
        RuntimeModelDPUInvariantFields * invariant;
        std::vector<RuntimeModelDPUVariantFields*> variant;
    };
}

#endif
