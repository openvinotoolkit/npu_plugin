// {% copyright %}

#ifndef SHARED_MODULES_SOFTMAX_H
#define SHARED_MODULES_SOFTMAX_H

#include "Op.h"

class Softmax: public Op
{
public:
    Softmax() = default;
    Softmax(t_MvTensorOpType /*op_type*/) : Op(kSoftMax) {
        _axis = 0;
    }

    void run(mv::tensor::Processor& mvtp,
            ::t_MvTensorMyriadResources& myriadRes,
            ::t_MvTensorDebugInfo& debugInfo) override;

    int32_t& axis() { return _axis; }
    Buffer input;
    Buffer output;

private:
    int32_t _axis;
};

#endif
