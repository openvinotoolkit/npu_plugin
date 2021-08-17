/*
* {% copyright %}
*/

#ifndef SHARED_MODULES_NEW_LAYER_H
#define SHARED_MODULES_NEW_LAYER_H

#include "Op.h"

#include "layers/param_pad.h"

class Pad: public Op
{
public:
    Pad() = default;
    Pad(t_MvTensorOpType /*op_type*/) : Op(kPad) {
        _pad0_begin = 0;
        _pad1_begin = 0;
        _pad2_begin = 0;
        _pad3_begin = 0;

        _pad0_end = 0;
        _pad1_end = 0;
        _pad2_end = 0;
        _pad3_end = 0;

        _padValue = 0.0f;
        _pad_mode = ePadMode::Constant;
    }

    virtual ~Pad() override;

    virtual void run(mv::tensor::Processor&,
                     t_MvTensorMyriadResources& myriadRes,
                     t_MvTensorDebugInfo& debugInfo) override;

    uint32_t& pad0_begin() { return _pad0_begin; }
    uint32_t& pad1_begin() { return _pad1_begin; }
    uint32_t& pad2_begin() { return _pad2_begin; }
    uint32_t& pad3_begin() { return _pad3_begin; }

    uint32_t& pad0_end() { return _pad0_end;}
    uint32_t& pad1_end() { return _pad1_end;}
    uint32_t& pad2_end() { return _pad2_end;}
    uint32_t& pad3_end() { return _pad3_end;}

    float& padValue() { return _padValue;}
    ePadMode& pad_mode() { return _pad_mode;}

    Buffer input;
    Buffer output;

private:
    uint32_t _pad0_begin;
    uint32_t _pad1_begin;
    uint32_t _pad2_begin;
    uint32_t _pad3_begin;

    uint32_t _pad0_end;
    uint32_t _pad1_end;
    uint32_t _pad2_end;
    uint32_t _pad3_end;

    float _padValue;
    ePadMode _pad_mode;
};

#endif //SHARED_MODULES_NEW_LAYER_H
