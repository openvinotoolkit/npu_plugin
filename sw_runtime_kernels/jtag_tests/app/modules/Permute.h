// {% copyright %}

#pragma once

#include "Op.h"

typedef struct
{
    int32_t order[MAX_DIMS];
    bool allow_permute_nd;
} t_PermuteLayerParams;

class Permute : public Op
{
public:
    Permute() : Op(kPermute) {};
    Permute(t_MvTensorOpType /*op_type*/) : Op(kPermute) {};
    virtual ~Permute() override;

    virtual void run(mv::tensor::Processor& mvtp,
            t_MvTensorMyriadResources& myriadRes,
            t_MvTensorDebugInfo& debugInfo) override;

    t_PermuteLayerParams ops;
    Buffer input;
    Buffer output;

    bool executeInTestingSystem = true;
};
