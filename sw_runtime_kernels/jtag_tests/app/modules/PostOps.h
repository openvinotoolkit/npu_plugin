// {% copyright %}

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
