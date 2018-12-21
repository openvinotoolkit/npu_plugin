#ifndef MV_RUNTIME_MODEL_DPU_TASK_
#define MV_RUNTIME_MODEL_DPU_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "include/mcm/compiler/runtime/tasks/runtime_model_nn_tensor_task.hpp"
#include "meta/schema/graphfile/nnNCE2_generated.h"

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
        std::vector<RuntimeModelPPELayerType> * ops;
        unsigned clampValueLow;
        unsigned clampValueHigh;
        unsigned ReLuNegSlope;
        unsigned ReLuPosSlope;
        unsigned pReLuAlpha;
    };

    flatbuffers::Offset<MVCNN::PPEFixedFunction> convertToFlatbuffer(RuntimeModelPPEFixedFunction * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreatePPEFixedFunctionDirect(
                    fbb,
                    ref->ops,
                    ref->clampValueLow,
                    ref->clampValueHigh);
    }

    std::vector<flatbuffers::Offset<MVCNN::PPEFixedFunction>> convertToFlatbuffer(std::vector<RuntimeModelPPEFixedFunction> * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::PPEFixedFunction>> toReturn;
        for(unsigned i = 0; i < ref->size(); ++i)
            toReturn.push_back(convertToFlatbuffer(ref->at(i), fbb));
        return toReturn;
    }

    struct RuntimeModelPPEGenericTask
    {
        RuntimeModelTensorReference * scaleData;
        std::vector<RuntimeModelPPEFixedFunction> * fixedFunction;
    };

    flatbuffers::Offset<MVCNN::PPETask> convertToFlatbuffer(RuntimeModelPPEGenericTask * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        MVCNN::CreatePPETaskDirect(fbb, convertToFlatbuffer(ref->scaleData, fbb), convertToFlatbuffer(ref->fixedFunction, fbb));
    }

    struct RuntimeModelDPUInvariantFields
    {
        RuntimeModelDPULayerType op;
        RuntimeModelPPEGenericTask * ppeTask;
        std::vector<RuntimeModelNNTask> * nnvShvTask;

        unsigned kernelH;
        unsigned kernelW;
        unsigned kernelStrideH;
        unsigned kernelStrideW;

        RuntimeModelTensorReference * inputData;
        RuntimeModelTensorReference * outputData;
        RuntimeModelTensorReference * weightsData;
        RuntimeModelTensorReference * biasData;
    };

    flatbuffers::Offset<MVCNN::NCEInvariantFields> convertToFlatbuffer(RuntimeModelDPUInvariantFields * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateNCEInvariantFieldsDirect(
                    fbb,
                    ref->op,
                    convertToFlatbuffer(ref->ppeTask, fbb),
                    convertToFlatbuffer(ref->nnvShvTask, fbb),
                    ref->kernelH,
                    ref->kernelW,
                    ref->kernelStrideH,
                    ref->kernelStrideW,
                    convertToFlatbuffer(ref->inputData, fbb),
                    convertToFlatbuffer(ref->outputData, fbb),
                    convertToFlatbuffer(ref->weightsData, fbb),
                    convertToFlatbuffer(ref->biasData, fbb));
    }

    struct RuntimeModelDPUVariantFields
    {
        unsigned clusterID;
        unsigned workloadID;
        RuntimeModelMPEMode mpeMode;

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

    flatbuffers::Offset<MVCNN::NCEVariantFields> convertToFlatbuffer(RuntimeModelDPUVariantFields * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return CreateNCEVariantFields(
            fbb,
            ref->clusterID,
            ref->workloadID,
            ref->mpeMode,
            ref->padLeft,
            ref->padRight,
            ref->padTop,
            ref->padBottom,
            ref->workload_start_X,
            ref->workload_start_Y,
            ref->workload_start_Z,
            ref->workload_end_X,
            ref->workload_end_Y,
            ref->workload_end_Z);
    }

    std::vector<flatbuffers::Offset<MVCNN::NCEVariantFields>> convertToFlatbuffer(std::vector<RuntimeModelDPUVariantFields> * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::NCEVariantFields>> toReturn;
        for(unsigned i = 0; i < ref->size(); ++i)
            toReturn.push_back(convertToFlatbuffer(ref->at(i), fbb));
        return toReturn;
    }


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
        std::vector<RuntimeModelDPUVariantFields> * variant;

        flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb)
        {
            return MVCNN::CreateNCE2TaskDirect(fbb, convertToFlatbuffer(invariant, fbb), convertToFlatbuffer(variant, fbb)).Union();
        }
    };
}

#endif
