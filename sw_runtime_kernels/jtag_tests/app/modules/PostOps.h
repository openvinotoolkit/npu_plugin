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

#ifndef SHARED_MODULES_POSTOPS_H_
#define SHARED_MODULES_POSTOPS_H_

#include "Op.h"

#include "layers/param_postops.h"

// kBias
// kClamp
// kElu
// kPRelu
// kPower
// kScale[Shift]
// kSigmoid
// kTanh
// k[Bias][Leaky]Relu
// kLog
// kExp
// kFloor
// kRound
// kErf
// kSwish
// kMish
// kGelu
// kLog

class PostOps: public Op
{
public:
    PostOps() = default;
    PostOps(t_MvTensorOpType op_type) : Op(op_type) {}
    virtual ~PostOps();

    virtual void run(mv::tensor::Processor& mvtp,
            t_MvTensorMyriadResources& myriadRes,
            t_MvTensorDebugInfo& debugInfo) override;

    union {
        float              opx;          // kElu, k[Bias][Leaky]Relu
        t_ClampLayerParams clampParams;  // kClamp
        t_PowerLayerParams powerParams;  // kPower
        t_SwishLayerParams swishParams;  // kSwish
        t_RoundLayerParams roundParams;  // kRound
    };

    OpTensor input;
    OpTensor output;
    OpTensor weights;  // kPRelu, kScale[Shift]
    OpTensor biases;   // kBias[[Leaky]Relu], kScaleShift

    unsigned paramsSize = 0;

    bool hasWeights = false;
    bool hasBiases = false;
    int axis = -1;

    bool executeInTestingSystem = true;

    // support for 3D/ND kernels differentiation (for separate kernel testing only)
#if defined(ICV_TESTS_SUPPORT)
    enum ForceKernel { ForceNone = 0, Force3D = 1, ForceND = 2 };
    ForceKernel forceKernel = ForceNone;
#endif // ICV_TESTS_SUPPORT
private:
//    void weightsBiasesSpecific(MVCNN::PostOpsParamsT *softLayerParamsValue, std::vector<Buffer>& inputs);
};

#endif /* SHARED_MODULES_POSTOPS_H_ */
