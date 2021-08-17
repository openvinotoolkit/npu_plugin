/*
* {% copyright %}
*/

#include "Pad.h"

#include "commonBuilder.hpp"
#include "upa_task_runner.hpp"

using namespace mv::tensor;

#define DEBUG_KERNELS 0
#if DEBUG_KERNELS
#include <stdio.h>
#define MVT_DPRINTF(...) printf(__VA_ARGS__)
#else
#define MVT_DPRINTF(...)
#endif

Pad::~Pad(){}

void Pad::run(mv::tensor::Processor& /*mvtp*/,
              t_MvTensorMyriadResources& myriadRes,
              t_MvTensorDebugInfo&)
{
    MVT_DPRINTF("ndims  = %ld\n", input.ndims);
    MVT_DPRINTF("order  = 0x%x\n", input.order);
    MVT_DPRINTF("dims.0 = %ld(%ld)\n", input.dims[0], input.strides[0]);
    MVT_DPRINTF("dims.1 = %ld(%ld)\n", input.dims[1], input.strides[1]);
    MVT_DPRINTF("dims.2 = %ld(%ld)\n", input.dims[2], input.strides[2]);
    MVT_DPRINTF("dims.3 = %ld(%ld)\n", input.dims[3], input.strides[3]);
    MVT_DPRINTF("mode   = %d\n", (int)(_pad_mode));

    std::unique_ptr<MVCNN::UPALayerTaskT> upaTask (new MVCNN::UPALayerTaskT());
    upaTask->softLayerParams.type = MVCNN::SoftwareLayerParams_PadParams;
    auto softLayerParamsValue = new MVCNN::PadParamsT();

    softLayerParamsValue->pads_begin = { _pad0_begin, _pad1_begin, _pad2_begin, _pad3_begin };
    softLayerParamsValue->pads_end = { _pad0_end, _pad1_end, _pad2_end, _pad3_end };

    softLayerParamsValue->padValue = padValue();
    softLayerParamsValue->pad_mode = (MVCNN::PadMode)(pad_mode());

    upaTask->softLayerParams.value = softLayerParamsValue;

    UPATaskRunner runner;
    mvTensorAssert(runner.enqueTask(std::move(upaTask), {input}, {output}, myriadRes.lastShave - myriadRes.firstShave + 1, &perfData), "pad layer run failed");
    mvTensorAssert(runner.dequeResult(), "pad layer run failed");
}
