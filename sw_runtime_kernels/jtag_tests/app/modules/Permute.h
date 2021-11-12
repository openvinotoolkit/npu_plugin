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
    OpTensor input;
    OpTensor output;

    bool executeInTestingSystem = true;
};
