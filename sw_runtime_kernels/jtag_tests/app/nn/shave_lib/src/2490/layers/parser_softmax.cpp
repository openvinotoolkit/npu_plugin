// {% copyright %}

#include "layers/parser_softmax.h"
#include "layers/param_softmax.h"
#include "layers/pre_softmax.h"
#include "layers/svuSLKernels_EP.h"
#include "tensor_gf_util.h"
//#include <nn_nce_lib_conversion_fbs.h>
#include <nn_cache.h>
#include <sw_layer.h>
#include <sw_shave_lib_common.h>
#include <limits.h>

namespace nn {
namespace shave_lib {

bool SoftmaxParser::parse(const MVCNN::UPALayerTask *task, Layer *layer) {
    if (task->softLayerParams_type() == MVCNN::SoftwareLayerParams::SoftwareLayerParams_SoftmaxParams) {
        const MVCNN::SoftmaxParams *gfParams = task->softLayerParams_as_SoftmaxParams();
        std::unique_ptr<t_MvSoftMaxParamNClasses> sp(new (std::nothrow) t_MvSoftMaxParamNClasses);
        parseInputs(task->inputs(), layer);
        parseOutputs(task->outputs(), layer);

        nnLog(MVLOG_DEBUG, "SoftmaxParser::parse input order 0x%lx, dtype %d, output order 0x%lx, dtype %d\n",
                layer->getInputs()[0].ndOrder, layer->getInputs()[0].dType,
                layer->getOutputs()[0].ndOrder, layer->getOutputs()[0].dType);
        bool success = false;
        NDDims indices = orderNDToIndices(((layer->getInputs())[0]).ndOrder, success);
        for (int i = 0; i < layer->getInputs()[0].ndims; i++) {
            nnLog(MVLOG_DEBUG, "SoftmaxParser::parse dim=%d) input: %d(%d), output: %d(%d), indices: %d\n", i,
                    layer->getInputs()[0].dims[i], layer->getInputs()[0].strides[i],
                    layer->getOutputs()[0].dims[i], layer->getOutputs()[0].strides[i], indices[i]);
        }

        TensorRef& inputData = layer->getInputs()[0];
        TensorRef& outputData = layer->getOutputs()[0];

        if (gfParams->axis() >= static_cast<unsigned int>(inputData.ndims)) {
            nnLog(MVLOG_ERROR, "Softmax: invalid axis %d", gfParams->axis());
            return false;
        }
        auto axis = /*(inputData.ndims - 1) - */gfParams->axis();
        sp->axis = indices[axis];
        nnLog(MVLOG_DEBUG, "SoftmaxParser::parse axis %d\n", sp->axis);
        if (layer->getInputs()[0].dims[sp->axis] == 1) {
            nnLog(MVLOG_ERROR, "Softmax on 1 element doesn't make sense (dim along the 'axis' equal 1)");
            return false;
        }

        memcpy_s(sp->in_dims,     MAX_ND_DIMS * sizeof(int32_t), inputData.dims,     MAX_ND_DIMS * sizeof(int32_t));
        memcpy_s(sp->in_strides,  MAX_ND_DIMS * sizeof(int32_t), inputData.strides,  MAX_ND_DIMS * sizeof(int32_t));
        memcpy_s(sp->out_strides, MAX_ND_DIMS * sizeof(int32_t), outputData.strides, MAX_ND_DIMS * sizeof(int32_t));
        sp->ndims = inputData.ndims;

        // excluding dim == 1 from dims
        for (int i = sp->ndims - 1; i >= 0; --i) {
            if (sp->ndims <= 1)
                break;
            if (sp->in_dims[i] == 1) {
                nnLog(MVLOG_DEBUG, "excluded: i %d, idim %d, istride %d, ostride %d, axis %d", i,
                        sp->in_dims[i], sp->in_strides[i], sp->out_strides[i], sp->axis);
                arrayElementExclude(sp->in_dims, i, sp->ndims);
                sp->ndims = arraysElementExclude(sp->in_strides, sp->out_strides, i, sp->ndims);
                sp->axis = (sp->axis > i) ? sp->axis - 1 : sp->axis;
                nnLog(MVLOG_DEBUG, ", new_axis %d, new ndims: %d\n", sp->axis, sp->ndims);
            }
        }

        if (sp->axis == 0 &&
                (sp->in_strides[0]  > static_cast<int32_t>(sizeof(fp16)) ||
                 sp->out_strides[0] > static_cast<int32_t>(sizeof(fp16)))) {
            arrayElementInclude(sp->in_dims, 0, 1, 1);
            arrayElementInclude(sp->in_strides,  0, sp->in_strides[0],  1);
            arrayElementInclude(sp->out_strides, 0, sp->out_strides[0], 1);
            sp->ndims++;
            sp->axis = 1;
        }

        if (sp->ndims < 3) { // works only with ndims >= 3 to simplicity
            for (int i = sp->ndims; i < 3; i++) {
                sp->in_strides[i] = sp->in_strides[sp->ndims - 1];
                sp->out_strides[i] = sp->out_strides[sp->ndims - 1];
                sp->in_dims[i] = 1;
            }
            sp->ndims = 3;
        }

        sp->axisDim = sp->in_dims[sp->axis];
        if (sp->axis) {
            sp->axisIStride = sp->in_strides[sp->axis];
            sp->axisOStride = sp->out_strides[sp->axis];
        } else {
            sp->axisIStride = sp->in_strides[1];
            sp->axisOStride = sp->out_strides[1];
        }

        arrayElementExclude(sp->in_dims, sp->axis, sp->ndims);
        sp->ndims = arraysElementExclude(sp->in_strides, sp->out_strides, sp->axis, sp->ndims);

        if (sp->axisDim > (int)(SHAVE_LIB_DATA_SIZE / 2 - 8 * sizeof(half) / sizeof(half))) {
            nnLog(MVLOG_ERROR, "CMX memory is not enough!");
            return false;
        }

        unsigned int id = getParamID(MVCNN::SoftwareLayerParams::SoftwareLayerParams_SoftmaxParams);
        cache::flush(*sp);

        layer->setParams(id, static_cast<LayerParams *>(sp.release()));
        layer->setPreamble(PREAMBLE_FUNC(preSoftmax));

        layer->setKernelEntry(KERNEL_FUNC(mvSoftMax));

        return true;
    }

    return false;
}
} // namespace shave_lib
} // namespace nn
