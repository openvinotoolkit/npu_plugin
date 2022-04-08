//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "PostOps.h"

#include "mvSubspaces.h"
#include <mvTensorDebug.h>

#include <math.h>

#include "shave_task_runner.hpp"

//#ifdef CONFIG_TARGET_SOC_3720
# include <Fp16Convert.h>
//#endif // CONFIG_TARGET_SOC_3720

#define DEBUG_KERNEL 0
#if DEBUG_KERNEL
# include <stdio.h>
# define MVT_DPRINTF(...) printf(__VA_ARGS__)
#else
# define MVT_DPRINTF(...) /* empty */
#endif

PostOps::~PostOps()
{
}

//void PostOps::weightsBiasesSpecific(MVCNN::PostOpsParamsT *softLayerParamsValue, std::vector<Buffer>& inputs) {
//    softLayerParamsValue->has_weights = hasWeights;
//    softLayerParamsValue->has_bias = hasBiases;
//
//    if ((hasWeights || hasBiases) && this->axis >= 0) {
//        softLayerParamsValue->axis = CommonFBFuilder::buildAxisIndex(this->axis, input.order);
//    }
//
//    inputs.push_back(input);
//    if (hasWeights)
//        inputs.push_back(weights);
//    if (hasBiases)
//        inputs.push_back(biases);
//}

void PostOps::run(mv::tensor::Processor&,
            t_MvTensorMyriadResources& myriadRes,
            t_MvTensorDebugInfo& /*debugInfo*/)
{
//    std::unique_ptr<MVCNN::UPALayerTaskT> upaTask (new MVCNN::UPALayerTaskT());
//    upaTask->softLayerParams.type = MVCNN::SoftwareLayerParams_PostOpsParams;
//    MVCNN::PostOpsParamsT *softLayerParamsValue = new MVCNN::PostOpsParamsT();

    std::vector<OpTensor> inputs;
//    weightsBiasesSpecific(softLayerParamsValue, inputs);
//#ifdef CONFIG_TARGET_SOC_3720
    /*if (this->executeInTestingSystem && (opType == kClamp))*/ {
        int32_t indices[subspace::MAX_DIMS] = {0, };
        const int total = subspace::getTotal(input.dims, input.ndims);
        for (int i = 0; i < total; ++i) {
            auto in = static_cast<const fp16*>(nn::element(input, indices));
            auto out = static_cast<fp16*>(nn::element(output, indices));
            *out = f32Tof16(std::max(clampParams.min, std::min(f16Tof32(*in), clampParams.max)));
            subspace::increment1Coord(indices, input.dims, input.ndims);
        }
    } //else {
//#endif // CONFIG_TARGET_SOC_3720
//
//    switch(opType)
//    {
//        case (kClamp):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_ClampParams;
//            auto params = std::unique_ptr<MVCNN::ClampParamsT>(new MVCNN::ClampParamsT());
//            params->min = clampParams.min;
//            params->max = clampParams.max;
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kElu):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_EluParams;
//            auto params = std::unique_ptr<MVCNN::EluParamsT>(new MVCNN::EluParamsT());
//            params->x = opx;
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kPower):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_PowerParams;
//            auto params = std::unique_ptr<MVCNN::PowerParamsT>(new MVCNN::PowerParamsT());
//            params->shift = powerParams.shift;
//            params->scale = powerParams.scale;
//            params->power = powerParams.power;
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kBiasLeakyRelu):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_BiasLeakyReluParams;
//            auto params = std::unique_ptr<MVCNN::BiasLeakyReluParamsT>(new MVCNN::BiasLeakyReluParamsT());
//            params->negative_slope = opx;
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kBiasRelu):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_BiasReluParams;
//            auto params = std::unique_ptr<MVCNN::BiasReluParamsT>(new MVCNN::BiasReluParamsT());
//            params->negative_slope = opx;
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kLeakyRelu):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_LeakyReluParams;
//            auto params = std::unique_ptr<MVCNN::LeakyReluParamsT>(new MVCNN::LeakyReluParamsT());
//            params->negative_slope = opx;
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kRelu):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_ReluParams;
//            softLayerParamsValue->nested_params.value = new MVCNN::ReluParamsT();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kPRelu):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_PReluParams;
//            softLayerParamsValue->nested_params.value = new MVCNN::PReluParamsT();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kSigmoidPostop):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_SigmoidParams;
//            softLayerParamsValue->nested_params.value = new MVCNN::SigmoidParamsT();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kTanh):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_TanhParams;
//            softLayerParamsValue->nested_params.value = new MVCNN::TanhParamsT();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kBias):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_BiasParams;
//            softLayerParamsValue->nested_params.value = new MVCNN::BiasParamsT();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kScale):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_ScaleParams;
//            softLayerParamsValue->nested_params.value = new MVCNN::ScaleParamsT();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kScaleShift):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_ScaleShiftParams;
//            softLayerParamsValue->nested_params.value = new MVCNN::ScaleShiftParamsT();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kHSwish):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_HSwishParams;
//            auto params = std::unique_ptr<MVCNN::HSwishParamsT>(new MVCNN::HSwishParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kSwish):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_SwishParams;
//            auto params = std::unique_ptr<MVCNN::SwishParamsT>(new MVCNN::SwishParamsT());
//            params->beta = swishParams.beta;
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kSoftPlus):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_SoftPlusParams;
//            auto params = std::unique_ptr<MVCNN::SoftPlusParamsT>(new MVCNN::SoftPlusParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kMish):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_MishParams;
//            auto params = std::unique_ptr<MVCNN::MishParamsT>(new MVCNN::MishParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kFloor):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_FloorParams;
//            auto params = std::unique_ptr<MVCNN::FloorParamsT>(new MVCNN::FloorParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kRound):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_RoundParams;
//            auto params = std::unique_ptr<MVCNN::RoundParamsT>(new MVCNN::RoundParamsT());
//            params->mode = static_cast<MVCNN::RoundMode>(roundParams.mode); // enum-to-enum: same int values
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kErf):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_ErfParams;
//            auto params = std::unique_ptr<MVCNN::ErfParamsT>(new MVCNN::ErfParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kCeiling):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_CeilingParams;
//            auto params = std::unique_ptr<MVCNN::CeilingParamsT>(new MVCNN::CeilingParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kGelu):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_GeluParams;
//            auto params = std::unique_ptr<MVCNN::GeluParamsT>(new MVCNN::GeluParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kLog):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_LogParams;
//            auto params = std::unique_ptr<MVCNN::LogParamsT>(new MVCNN::LogParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        case (kExp):
//        {
//            softLayerParamsValue->nested_params.type = MVCNN::PostOpsNestedParams::PostOpsNestedParams_ExpParams;
//            auto params = std::unique_ptr<MVCNN::ExpParamsT>(new MVCNN::ExpParamsT());
//            softLayerParamsValue->nested_params.value = params.release();
//            upaTask->softLayerParams.value = softLayerParamsValue;
//            break;
//        }
//        default:
//            mvTensorAssert(false && "PostOp type is not supported");
//            break;
//    }

//    std::vector<Buffer> inputs;
    inputs.push_back(input);
    if (hasWeights)
        inputs.push_back(weights);
    if (hasBiases)
        inputs.push_back(biases);

//    UPATaskRunner runner;
//    mvTensorAssert(runner.enqueTask(std::move(upaTask), inputs, {output}, myriadRes.lastShave - myriadRes.firstShave + 1, &perfData), "Postop layer run failed");
//    mvTensorAssert(runner.dequeResult(), "Postop layer run failed");

//#ifdef CONFIG_TARGET_SOC_3720
//    }
//#endif // CONFIG_TARGET_SOC_3720

}
