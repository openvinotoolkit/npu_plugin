// {% copyright %}

#include "commonBuilder.hpp"

#include "Softmax.h"
#include "upa_task_runner.hpp"
#include "layers/param_softmax.h"

#include <memory>

#define DEBUG_KERNEL 0
#if DEBUG_KERNEL
#include <stdio.h>
#define MVT_DPRINTF(...) printf(__VA_ARGS__)
#else
#define MVT_DPRINTF(...)
#endif

using namespace nn::shave_lib;

void Softmax::run(mv::tensor::Processor& ,
                  t_MvTensorMyriadResources& myriadRes,
                  t_MvTensorDebugInfo&) {

    MVT_DPRINTF("ndims  = %ld\n", input.ndims);
    MVT_DPRINTF("order  = 0x%x\n", input.order);
    MVT_DPRINTF("dims.0 = %ld(%ld)\n", input.dims[0], input.strides[0]);
    MVT_DPRINTF("dims.1 = %ld(%ld)\n", input.dims[1], input.strides[1]);
    MVT_DPRINTF("dims.2 = %ld(%ld)\n", input.dims[2], input.strides[2]);
    MVT_DPRINTF("dims.3 = %ld(%ld)\n", input.dims[3], input.strides[3]);
    MVT_DPRINTF("axis   = %ld(%ld)\n", _axis);

    mvTensorAssert(input.dims[_axis] > 1, "Softmax on less then 2 elements does not make sense!");

    // ---- KMB infra

    std::unique_ptr<MVCNN::UPALayerTaskT> upaTask (new MVCNN::UPALayerTaskT());
    upaTask->softLayerParams.type = MVCNN::SoftwareLayerParams_SoftmaxParams;
    auto softLayerParamsValue = new MVCNN::SoftmaxParamsT();

    softLayerParamsValue->axis = CommonFBFuilder::buildAxisIndex(_axis, input.order);
    upaTask->softLayerParams.value = softLayerParamsValue;

    UPATaskRunner runner;
    mvTensorAssert(runner.enqueTask(std::move(upaTask), {input}, {output}, myriadRes.lastShave - myriadRes.firstShave + 1, &perfData), "softmax layer run failed");
    mvTensorAssert(runner.dequeResult(), "softmax layer run failed");
}
