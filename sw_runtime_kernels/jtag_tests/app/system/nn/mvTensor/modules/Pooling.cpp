// {% copyright %}

#include "Pooling.h"
#include "commonBuilder.hpp"
#include "upa_task_runner.hpp"
#include <memory>
#include "mvTensorUtil.h"
#include <nn_log.h>

#define MAX(_a, _b) ((_a)>(_b)?(_a):(_b))

Pooling::~Pooling() {
}

void Pooling::run(mv::tensor::Processor& /*mvtp*/,
        t_MvTensorMyriadResources& myriadRes,
        t_MvTensorDebugInfo& /*debugInfo*/) {
    nnLog(MVLOG_DEBUG, "Pooling::run\n");
    orderToIndices(input.order, inIndices);
    orderToIndices(output.order, outIndices);

    int rPadX, bPadY;
    calcPad(input, output, rPadX, bPadY);

    std::unique_ptr<MVCNN::UPALayerTaskT> upaTask(new MVCNN::UPALayerTaskT);
    upaTask->softLayerParams.type = MVCNN::SoftwareLayerParams_PoolingParams;

    std::unique_ptr<MVCNN::PoolingParamsT> params(new MVCNN::PoolingParamsT);

    params->kernel.reset(new MVCNN::order3(radixX, radixY, 1));
    params->pool_method = pool_method;
    params->strides.reset(new MVCNN::order3(radixStrideX, radixStrideY, 1));
    params->pads_begin.reset(new  MVCNN::order3(padX, padY, 1));
    params->pads_end.reset(new  MVCNN::order3(rPadX, bPadY, 1));
    params->exclude_pad = (ops.excludePad) ? "true" : "false";

    upaTask->softLayerParams.value = params.release();

    // esmirno - avoid parser runner direct usage
    UPATaskRunner runner;

    mvTensorAssert(
        runner.enqueTask(std::move(upaTask), {input}, {output}, myriadRes.lastShave - myriadRes.firstShave + 1, &perfData),
        "Pooling layer run failed");

    mvTensorAssert(
        runner.dequeResult(),
        "Pooling layer run failed");
}

void Pooling::calcPad(const Buffer& input, const Buffer& output, int& rpad_x, int& bpad_y) {
    int IW = input.dims[inIndices[0]];
    int IH = input.dims[inIndices[1]];
    int OW = output.dims[outIndices[0]];
    int OH = output.dims[outIndices[1]];

    nnLog(MVLOG_DEBUG, "Test: IC %d, OC %d, iorder 0x%x, oorder  0x%x\n",
            input.dims[inIndices[2]], output.dims[inIndices[2]], input.order, output.order);
    rpad_x = mv::tensor::util::convPoolSizesRPadBySizeOutput(IW, OW,
            radixX, radixStrideX, padX);
    nnLog(MVLOG_DEBUG, "Test W: IW %d, OW %d, radixX %d, StrideX %d, padX %d, rPadX %d\n",
            IW, OW, radixX, radixStrideX, padX, rpad_x);

    bpad_y = mv::tensor::util::convPoolSizesRPadBySizeOutput(IH, OH,
            radixY, radixStrideY, padY);
    nnLog(MVLOG_DEBUG, "Test H: IH %d, OH %d, radixY %d, StrideY %d, padY %d, bPadY %d\n",
            IH, OH, radixY, radixStrideY, padY, bpad_y);

    rpad_x = MAX(rpad_x, 0);
    bpad_y = MAX(bpad_y, 0);
}
