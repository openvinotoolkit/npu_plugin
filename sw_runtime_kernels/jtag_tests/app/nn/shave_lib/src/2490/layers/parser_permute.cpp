/*
* {% copyright %}
*/
#include "layers/parser_permute.h"
#include "layers/param_permute.h"
#include "layers/pre_permute.h"
#include "layers/svuSLKernels_EP.h"
#include "sw_layer.h"
#include "sw_tensor_ref.h"
#include "tensor_gf_util.h"
#include "tensor_util.h"

#include <nn_cache.h>
#include <nn_log.h>

#define PERM_ORDER_123 0x123
#define PERM_ORDER_132 0x132
#define PERM_ORDER_213 0x213
#define PERM_ORDER_231 0x231
#define PERM_ORDER_312 0x312
#define PERM_ORDER_321 0x321

namespace nn {
namespace shave_lib {

// This function is a look-up table which maps the input permute order (which
// is with respect to (x,y,z) to the storage order used by the tensor.
//
// For example, some parameters from the input blob:
//
//     dimensions_in (z,y,x) = {4,3,2}
//     permute_order (x,y,z) = {1,2,0}
//     storage_order         = HWC
//
// Steps:
//
//     1. Re-order dimensions_in to (x,y,z) (i.e. the format expected by the
//        permute order).
//
//        --> dimensions_in  (x,y,z) = {2,3,4}
//
//     2. Determine the output dimensions, using the permute order from the blob
//        (which is with respect to (x,y,z)).
//
//            dimensions_in  (x,y,z) = {2,3,4}
//            permute_order  (x,y,z) = {1,2,0}
//        --> dimensions_out (x,y,z) = {3,4,2}
//
//     3. Re-order the output dimensions to the storage order used by the tensor
//        (HWC order, i.e. (y,x,z)).
//
//        --> dimensions_out (y,x,z) = {4,3,2}
//
//     4. Determine the permute order with respect to the storage order used,
//        based on the tensor dimensions before and after the permutation.
//
//            dimensions_in  (y,x,z) = {3,2,4}
//            dimensions_out (y,x,z) = {4,3,2}
//        --> permute_order  (y,x,z) = {2,0,1}
//
//
// This function maps all possible permute orders in (x,y,z) format to their
// storage order format in the same manner as above, implemented as a look-up
// table.
//
// This behaviour may be taken care of by mcmCompiler in the future.
//
static bool permuteOrderToStorageOrder(int32_t permute_order_out[], const uint32_t permute_order_xyz, const NDOrder storageOrder) {
    switch (storageOrder) {
        case ND_NHWC:
            switch(permute_order_xyz) {
                case PERM_ORDER_123:
                    permute_order_out[0] = 2;
                    permute_order_out[1] = 0;
                    permute_order_out[2] = 1;
                    break;
                case PERM_ORDER_132:
                    permute_order_out[0] = 1;
                    permute_order_out[1] = 0;
                    permute_order_out[2] = 2;
                    break;
                case PERM_ORDER_213:
                    permute_order_out[0] = 0;
                    permute_order_out[1] = 2;
                    permute_order_out[2] = 1;
                    break;
                case PERM_ORDER_231:
                    permute_order_out[0] = 2;
                    permute_order_out[1] = 0;
                    permute_order_out[2] = 1;
                    break;
                case PERM_ORDER_312:
                    permute_order_out[0] = 1;
                    permute_order_out[1] = 2;
                    permute_order_out[2] = 0;
                    break;
                case PERM_ORDER_321:
                    permute_order_out[0] = 2;
                    permute_order_out[1] = 1;
                    permute_order_out[2] = 0;
                    break;
                default:
                    return false;
            }
            break;
        case ND_NCHW:
            switch(permute_order_xyz) {
                case PERM_ORDER_123:
                    permute_order_out[0] = 0;
                    permute_order_out[1] = 1;
                    permute_order_out[2] = 2;
                    break;
                case PERM_ORDER_132:
                    permute_order_out[0] = 1;
                    permute_order_out[1] = 0;
                    permute_order_out[2] = 2;
                    break;
                case PERM_ORDER_213:
                    permute_order_out[0] = 0;
                    permute_order_out[1] = 2;
                    permute_order_out[2] = 1;
                    break;
                case PERM_ORDER_231:
                    permute_order_out[0] = 2;
                    permute_order_out[1] = 0;
                    permute_order_out[2] = 1;
                    break;
                case PERM_ORDER_312:
                    permute_order_out[0] = 1;
                    permute_order_out[1] = 2;
                    permute_order_out[2] = 0;
                    break;
                case PERM_ORDER_321:
                    permute_order_out[0] = 2;
                    permute_order_out[1] = 1;
                    permute_order_out[2] = 0;
                    break;
                default:
                    return false;
            }
            break;
        default:
            return false;
    }
    return true;
}

static NDDims logicalDimsPermutationToStorageDimsPermutation(
        TensorRef &inputData,
        TensorRef &outputData,
        const s32 logicalPermutation[]) {
    NDDims  memory_order_permutation;

    s32 tmp[MAX_ND_DIMS];
    std::memset(tmp, 0x0, MAX_ND_DIMS * sizeof(tmp[0]));
    bool success = false;
    NDDims orderPerm = subspace::orderNDToPermutation(outputData.ndOrder, success);
    memory_order_permutation.resize(orderPerm.ndims());
    NDDims orderInd = subspace::orderNDToIndices(inputData.ndOrder, success);

    /*
    orderPerm - (Op) permutation from logicalDims to memoryDims (logicalDims --> (orderPerm) --> memoryDims)
                or memoryDims = Op(logicalDims)
    orderInd  - (Oi) inverse permutation from logicalDims to memoryDims i.e.
                permutation from memoryDims to logicalDims (memoryDims --> (orderInd) --> logicalDims)
                or logicalDims = Oi(memoryDims)
    logicalPermutation - (P) permutation of logical dims (logOutDims = P(logInDims)):
    memory_order_permutation - (Pm) corresponding permutation of memory dims (memOutDims = Ps(memInDims)),
    then:
    logOutDims = P(logInDims)) ->
    Oi(memOutDims) = P(Oi(memInDims)) -> applying Op permutation (inverse for Oi) to both equality sides ->
    memOutDims = Op(P(Oi(memInDims)))
    then, memory_order_permutation can be found as:
    Pm = Op(P(Oi))
    */
    subspace::permuteArray(orderInd.data(), logicalPermutation, tmp, inputData.ndims);
    subspace::permuteArray(tmp, orderPerm.data(), memory_order_permutation.data(), inputData.ndims);
    return memory_order_permutation;
}

void layerCleanupPermute(const LayerParams *params) {
    auto plp = static_cast<const PermuteParams *>(params);
    nn::memory::cache_aligned_free(plp->parsedPerm);
}

// common part for PermuteParser and PermuteNDParser
static bool configKernelCommon(TensorRef &inputData,
                               TensorRef &outputData,
                               const s32 permutation[],
                               const u32 bpp,
                               const unsigned int param_id,
                               Layer *layer) {
    memory::cache_aligned_unique_ptr<PermForDMA> parsedPerm(new (memory::cache_aligned) PermForDMA);
    if (parsedPerm == nullptr) {
        nnLog(MVLOG_ERROR, "Can not allocate PermForDMA structure");
        return false;
    }

    s32  memory_order_permutation[MAX_ND_DIMS];
    std::memset(memory_order_permutation, 0x0, MAX_ND_DIMS * sizeof(memory_order_permutation[0]));
    s32  memory_order_indices[MAX_ND_DIMS];
    std::memset(memory_order_indices, 0x0, MAX_ND_DIMS * sizeof(memory_order_indices[0]));
    s32  tmp[MAX_ND_DIMS];
    std::memset(tmp, 0x0, MAX_ND_DIMS * sizeof(tmp[0]));

    permuteArray(inputData.dims, permutation, tmp, inputData.ndims);

    unsigned int elements = (inputData.ndims > 0) ? 1 : 0;
    for(int i = 0; i < inputData.ndims; ++i)
    {
        if (tmp[i] != outputData.dims[i]) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE] Sizes of input and output (permuted) tensors should correspond");
            return false;
        }
        elements *= inputData.dims[i];
        memory_order_permutation[i] = permutation[i];
    }

    // Do nothing if input tensor is empty
    if (elements == 0) {
        nnLog(MVLOG_ERROR, "[PARSER PERMUTE] Empty input");
        return false;
    }

    // Matrix transpose case
    // at this point it's impossible to determine whether there are enough CMX memory for transpose case
    // FIXME uncomment the following case
    /*
    bool matrix_transpose_with_batch = (permutation[0] == 1 && permutation[1] == 0 && permutation[2] == 2 && inputData.ndims == 3) ||
                                       (permutation[0] == 1 && permutation[1] == 0 && permutation[2] == 2 && permutation[3] == 3 && inputData.dims[3] == 1 && inputData.ndims == 4);
    bool matrix_transpose = (permutation[0] == 1 && permutation[1] == 0 && inputData.ndims == 2);

    if ((matrix_transpose_with_batch || matrix_transpose) && (bpp == sizeof(half)))
    {
        int first_shave = 0;
        int last_shave  = 0;

        int batch_index = 2;

        int batch_nums = (inputData.ndims == 2) ? 1 : inputData.dims[batch_index];
        int height = inputData.dims[1];
        bool height_tiling = (height > batch_nums * 8);

        u32 nshaves = last_shave - first_shave + 1;
        u32 batch_step  = height_tiling ? (batch_nums) : (batch_nums + (nshaves - 1)) / nshaves;
        u32 height_step = height_tiling ? (height + (nshaves - 1)) / nshaves : height;

        t_PermTransposeParam * params = new (std::nothrow) t_PermTransposeParam();

        params->width = inputData.dims[0];

        params->stride_in_line = inputData.strides[1];
        params->stride_in_batch = inputData.strides[batch_index];

        params->stride_out_line = outputData.strides[1];
        params->stride_out_batch = outputData.strides[batch_index];

        params->batch0 = height_tiling ? 0 : batch_step * (shave_idx - first_shave);
        params->batch1 = MIN((int)(batch_step * (shave_idx - first_shave + 1)), batch_nums);

        params->height0 = height_tiling ? height_step * (shave_idx - first_shave) : 0;
        params->height1 = MIN((int)(height_step * (shave_idx - first_shave + 1)), height);

        layer->setParams(param_id, static_cast<LayerParams *>(params));

        cache::flush(pp, sizeof(t_PermTransposeParam));

        layer->setPreamble(PREAMBLE_FUNC(preMvPermuteTranspose));
        layer->setKernelEntry(KERNEL_FUNC(mvPermuteTranspose));

        return true;
    }
    */

    // excluding dim == 1 from dims
    for(int i = inputData.ndims - 1; i >= 0; --i)
    {
        if(inputData.dims[memory_order_permutation[i]] == 1)
        {
            // dimension exclusion
            subspace::arrayElementExclude(inputData.dims, memory_order_permutation[i], inputData.ndims);
            subspace::arrayElementExclude(inputData.strides, memory_order_permutation[i], inputData.ndims);
            subspace::arrayElementExclude(outputData.dims, i, outputData.ndims);
            outputData.ndims = subspace::arrayElementExclude(outputData.strides, i, outputData.ndims);

            for(int j = 0; j < inputData.ndims; ++ j)
            {
                memory_order_permutation[j]
                          = (memory_order_permutation[j] > memory_order_permutation[i]) ?
                             memory_order_permutation[j] - 1 : memory_order_permutation[j];
            }

            inputData.ndims = subspace::arrayElementExclude(memory_order_permutation, i, inputData.ndims);
            if(inputData.ndims <= 1) break;
        }
    }

    if(inputData.ndims == 1)
    {
        PermuteParams1D * p_permuteParams1D = new (std::nothrow) PermuteParams1D();
        p_permuteParams1D->bpp = bpp;
        p_permuteParams1D->inWidth = inputData.dims[0];
        p_permuteParams1D->inWidthStride = inputData.strides[0];
        p_permuteParams1D->outWidthStride = outputData.strides[0];

        layer->setParams(param_id, static_cast<LayerParams *>(p_permuteParams1D));

        cache::flush(p_permuteParams1D, sizeof(PermuteParams1D));

        layer->setPreamble(PREAMBLE_FUNC(prePermute1D));
        layer->setKernelEntry(nullptr);
        return true;
    }

    if(inputData.ndims < 3)
    {
        // expand up to 3 dimensions to simplify further processing
        for(int i = inputData.ndims; i < 3; i++)
        {
            memory_order_permutation[i] = i;
            inputData.strides[i] = inputData.strides[inputData.ndims - 1];
            outputData.strides[i] = outputData.strides[inputData.ndims - 1];
            inputData.dims[i]  = 1;
            outputData.dims[i] = 1;
        }
        inputData.ndims = 3;
        outputData.ndims = 3;
    }

    for(int i = 0; i < inputData.ndims; ++ i)
    {
        memory_order_indices[memory_order_permutation[i]] = i;
    }

    uint32_t ind1_in, ind1_out;

    if(memory_order_permutation[0] != 0)
    {
        ind1_in = memory_order_permutation[0];
        ind1_out = memory_order_indices[0];
        parsedPerm->transpose = true;
    }
    else
    {
        ind1_in = memory_order_permutation[1];
        ind1_out = 1;
        parsedPerm->transpose = false;
    }
    parsedPerm->dims_in[0] = inputData.dims[0];
    parsedPerm->dims_out[0] = outputData.dims[0];
    parsedPerm->strides_in[0] = inputData.strides[0];
    parsedPerm->strides_out[0] = outputData.strides[0];
    parsedPerm->dims_in[1] = inputData.dims[ind1_in];
    parsedPerm->strides_in[1] = inputData.strides[ind1_in];
    parsedPerm->dims_out[1] = outputData.dims[ind1_out];
    parsedPerm->strides_out[1] = outputData.strides[ind1_out];

    parsedPerm->ndims = inputData.ndims;
    std::memset(parsedPerm->slice_sizes, 0x0, (MAX_ND_DIMS-2) * sizeof(parsedPerm->slice_sizes[0]));

    int n_of_slices = 1;
    unsigned int i = 1;
    for(; i < ind1_out; ++i)
    {
        parsedPerm->dims_out[i + 1] = outputData.dims[i];
        parsedPerm->strides_out[i + 1] = outputData.strides[i];
        parsedPerm->dims_in[i + 1] = inputData.dims[memory_order_permutation[i]];
        parsedPerm->strides_in[i + 1] = inputData.strides[memory_order_permutation[i]];
        parsedPerm->slice_sizes[i - 1] = n_of_slices;
        n_of_slices *= outputData.dims[i];
    }
    i++;
    for(; i < parsedPerm->ndims; ++i)
    {
        parsedPerm->dims_out[i] = outputData.dims[i];
        parsedPerm->strides_out[i] = outputData.strides[i];
        parsedPerm->dims_in[i] = inputData.dims[memory_order_permutation[i]];
        parsedPerm->strides_in[i] = inputData.strides[memory_order_permutation[i]];
        parsedPerm->slice_sizes[i - 2] = n_of_slices;
        n_of_slices *= outputData.dims[i];
    }

    PermuteParams * p_permuteParams = new (std::nothrow) PermuteParams();
    p_permuteParams->run_mv_transpose = false;
    p_permuteParams->bpp = bpp;
    p_permuteParams->n_of_slices = n_of_slices;
    p_permuteParams->maxInnerDims = std::max(static_cast<int>(parsedPerm->dims_in[1]), static_cast<int>(parsedPerm->dims_in[0]));
    cache::flush(*parsedPerm);
    p_permuteParams->parsedPerm = parsedPerm.release();

    layer->setParams(param_id, static_cast<LayerParams *>(p_permuteParams));

    cache::flush(p_permuteParams, sizeof(PermuteParams));

    layer->setPreamble(PREAMBLE_FUNC(prePermute));
    layer->setKernelEntry(KERNEL_FUNC(nnPermute));
    layer->setLayerCleanup(&layerCleanupPermute);

    return true;
}

bool PermuteParser::parse(const MVCNN::UPALayerTask *task, Layer *layer) {
    if (task->softLayerParams_type() == MVCNN::SoftwareLayerParams::SoftwareLayerParams_PermuteParams) {

        if (!parseUPATensors(task, layer)) {
            nnLog(MVLOG_ERROR, "Permute : input/output tensors parsing fails.");
            return false;
        }
        auto inputs = layer->getInputs();
        auto outputs = layer->getOutputs();
        auto& inputData = inputs[0];
        auto& outputData = outputs[0];
        const auto bpp = nn::getBpp(inputData.dType);
        if (bpp != 1 && bpp != 2 && bpp != 4) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE] Unsupported data type: bits per pixel: %d", bpp);
            return false;
        }
        const auto out_bpp = nn::getBpp(outputData.dType);
        if (out_bpp != bpp) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE] Input data type %d does not match output data type %d", bpp, out_bpp);
            return false;
        }

        // TODO currently permutation was only tested with NCHW and NHWC output layouts
        // rework permuteOrderToStorageOrder to support all layout combinations
        if (outputData.ndOrder != ND_NCHW && outputData.ndOrder != ND_NHWC) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE] outputData.order %x is not supported", outputData.ndOrder);
            return false;
        }

        if (inputData.ndOrder != outputData.ndOrder) {
            nnLog(MVLOG_INFO, "[PARSER PERMUTE] inputData.ndOrder %llx does not match outputData.ndOrder %llx", inputData.ndOrder, outputData.ndOrder);
        }

        const auto permuteOrder = task->softLayerParams_as_PermuteParams()->permute_order();
        if (permuteOrder == nullptr) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE] permute_order is a null-pointer");
            return false;
        }
        auto permutation_x = static_cast<int32_t>(permuteOrder->x());
        auto permutation_y = static_cast<int32_t>(permuteOrder->y());
        auto permutation_z = static_cast<int32_t>(permuteOrder->z());
        // permutation is reversed here because MCM schema implies reversed memory ordered notation
        // (order, opposite to caffe framework)
        s32 permutation[] = { 2 - permutation_z, 2 - permutation_y, 2 - permutation_x, 3 };
        NDDims permOrderArr;
        permOrderArr.resize(3);
        permOrderArr[0] = permutation_z; permOrderArr[1] = permutation_y, permOrderArr[2] = permutation_x;
        NDOrder permOrderHex = subspace::permutationToOrderND(permOrderArr);
        if (!permuteOrderToStorageOrder(permutation, permOrderHex, outputData.ndOrder)) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE] cannot match permute order %x and storage order %x", permOrderHex, outputData.ndOrder);
            return false;
        }

        unsigned int param_id = getParamID(MVCNN::SoftwareLayerParams::SoftwareLayerParams_PermuteParams);
        return configKernelCommon(inputData, outputData, permutation, bpp, param_id, layer);
    }

    return false;
}

bool PermuteNDParser::parse(const MVCNN::UPALayerTask *task, Layer *layer) {
    if (task->softLayerParams_type() == MVCNN::SoftwareLayerParams::SoftwareLayerParams_PermuteNDParams) {
        parseUPATensors(task, layer);
        auto inputs = layer->getInputs();
        auto outputs = layer->getOutputs();
        auto& inputData = inputs[0];
        auto& outputData = outputs[0];
        const auto bpp = nn::getBpp(inputData.dType);
        if (bpp != 1 && bpp != 2 && bpp != 4) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE ND] Unsupported data type: bits per pixel: %d", bpp);
            return false;
        }
        const auto out_bpp = nn::getBpp(outputData.dType);
        if (out_bpp != bpp) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE ND] Input data type %d does not match output data type %d", bpp, out_bpp);
            return false;
        }

        const flatbuffers::Vector<int32_t> * permuteOrder = task->softLayerParams_as_PermuteNDParams()->permute_nd_order();
        if (permuteOrder == nullptr) {
            nnLog(MVLOG_ERROR, "[PARSER PERMUTE ND] permute_order is a null-pointer");
            return false;
        }

        s32 logicalPermutation[MAX_ND_DIMS];
        std::memset(logicalPermutation, 0x0, MAX_ND_DIMS * sizeof(logicalPermutation[0]));
        for (size_t dimIdx = 0; dimIdx < permuteOrder->size(); dimIdx++) {
            logicalPermutation[dimIdx] = permuteOrder->Get(dimIdx);
        }
        NDDims permutation = logicalDimsPermutationToStorageDimsPermutation(inputData, outputData, logicalPermutation);

        unsigned int param_id = getParamID(MVCNN::SoftwareLayerParams::SoftwareLayerParams_PermuteNDParams);
        return configKernelCommon(inputData, outputData, permutation.data(), bpp, param_id, layer);
    }

    return false;
}
} // namespace shave_lib
} // namespace nn
