// {% copyright %}

#include <layers/pre_postops.h>
#include <layers/param_postops.h>
#include <sw_layer.h>
#include <sw_shave_res_manager.h>

//#define ASYNC_PIPELINE /* TODO: fully remove async code, if async pipelining isn't supported */

namespace nn {
namespace shave_lib {

void prePostOpsHWC(const LayerParams *layerParams, ShaveResourceManager *resMgr) {
    const PostOpsParams *ddrParams = static_cast<const PostOpsParams *>(layerParams);

    unsigned int numShaves = resMgr->getMaximumShaves();

    auto input_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(0));
    const half* weights_data = nullptr;
    const half* biases_data = nullptr;

    int input_idx = 1;
    if (ddrParams->has_weights) {
        weights_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }
    if (ddrParams->has_biases) {
        biases_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }

    auto output_data = reinterpret_cast<half*>(resMgr->getAbsoluteOutputAddr(0));

    auto res = resMgr->requestShaves(numShaves);

    int first_shave = 0;
    int last_shave = numShaves - 1;

    u32 no_shaves = last_shave - first_shave + 1;
    u32 h_per_shave = ddrParams->height / no_shaves;
    u32 h_per_shave_remainder = ddrParams->height % no_shaves;
    s32 in_offset = 0;
    s32 out_offset = 0;

    for(int shave_i = first_shave; shave_i <= last_shave; ++shave_i)
    {
        const ShaveResource &sh = res[shave_i];
        resMgr->setupShaveForKernel(sh);
        t_HWCPostOps3DParams *postOpsParams = resMgr->getParams<t_HWCPostOps3DParams>(sh);

        *postOpsParams = *static_cast<const t_HWCPostOps3DParams*>(ddrParams);

        postOpsParams->input        = (input_data  + in_offset);
        postOpsParams->output       = (output_data + out_offset);
        postOpsParams->weights      = weights_data;
        postOpsParams->bias         = biases_data;
        postOpsParams->height       = h_per_shave;

        // Distribute one line of width to the first h_per_shave_remainder shave.
        in_offset += h_per_shave * ddrParams->in_step;
        out_offset += h_per_shave * ddrParams->out_step;

        if(h_per_shave_remainder != 0)
        {
            postOpsParams->height += 1;
            in_offset += ddrParams->in_step;
            out_offset += ddrParams->out_step;
            --h_per_shave_remainder;
        }

        resMgr->updateLayerParams(sh, postOpsParams);

#if defined(__leon_rt__) || defined(__leon__)
        rtems_cache_flush_multiple_data_lines(postOpsParams, sizeof(t_HWCPostOps3DParams));
#endif
    }
}

void prePostOpsCHW(const LayerParams *layerParams, ShaveResourceManager *resMgr) {
    const PostOpsParams *ddrParams = static_cast<const PostOpsParams *>(layerParams);
    unsigned int numShaves = resMgr->getMaximumShaves();

    auto input_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(0));
    const half* weights_data = nullptr;
    const half* biases_data = nullptr;

    int input_idx = 1;
    if (ddrParams->has_weights) {
        weights_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }
    if (ddrParams->has_biases) {
        biases_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }

    auto output_data = reinterpret_cast<half*>(resMgr->getAbsoluteOutputAddr(0));

    auto res = resMgr->requestShaves(numShaves);

    int first_shave = 0;
    int last_shave = numShaves - 1;

    u32 channels = ddrParams->channels;

    u32 used_shaves = std::min(static_cast<u32>(last_shave - first_shave + 1), channels);
    last_shave = first_shave + used_shaves - 1;

    u32 channels_per_shave = channels / used_shaves;
    u32 channels_per_shave_remainder = channels % used_shaves;
    s32 in_offset = 0;
    s32 out_offset = 0;
    s32 channels_offset = 0;

    for (int shave_i = first_shave; shave_i <= last_shave; ++shave_i)
    {
        const ShaveResource &sh = res[shave_i];
        resMgr->setupShaveForKernel(sh);
        t_CHWPostOps3DParams *postOpsParams = resMgr->getParams<t_CHWPostOps3DParams>(sh);
        *postOpsParams = *static_cast<const t_CHWPostOps3DParams*>(ddrParams);

        postOpsParams->input        = (input_data  + in_offset);
        postOpsParams->output       = (output_data + out_offset);
        postOpsParams->weights      = weights_data ? (weights_data + channels_offset) : weights_data;
        postOpsParams->bias         = biases_data ? (biases_data + channels_offset) : biases_data;

        u32 channels_extra = (channels_per_shave_remainder != 0) ? 1 : 0;
        u32 channels_used = channels_per_shave + channels_extra;

        postOpsParams->channels     = channels_used;

        in_offset += ddrParams->in_step * ddrParams->height * channels_used;
        out_offset += ddrParams->out_step * ddrParams->height * channels_used;
        channels_offset += channels_used;
        channels_per_shave_remainder -= channels_extra;

        resMgr->updateLayerParams(sh, postOpsParams);
#if defined(__leon_rt__) || defined(__leon__)
        rtems_cache_flush_multiple_data_lines(postOpsParams, sizeof(t_CHWPostOps3DParams));
#endif
    }
}

void prePostOpsHCW(const LayerParams *layerParams, ShaveResourceManager *resMgr) {
    const PostOpsParams *ddrParams = static_cast<const PostOpsParams *>(layerParams);
    unsigned int numShaves = 1;
    //TODO: enable multi shaves, now it works incorrect for Relu family - S#48990
    //unsigned int numShaves = resMgr->getMaximumShaves();

    auto input_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(0));
    const half* weights_data = nullptr;
    const half* biases_data = nullptr;

    int input_idx = 1;
    if (ddrParams->has_weights) {
        weights_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }
    if (ddrParams->has_biases) {
        biases_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }

    auto output_data = reinterpret_cast<half*>(resMgr->getAbsoluteOutputAddr(0));

    auto res = resMgr->requestShaves(numShaves);

    int first_shave = 0;
    int last_shave = numShaves - 1;

    u32 no_shaves =  last_shave - first_shave + 1;
    u32 h_per_shave = ddrParams->height / no_shaves;
    u32 h_per_shave_remainder = ddrParams->height % no_shaves;
    s32 in_out_offset = 0, start_line = 0;

    for(int shave_i = first_shave; shave_i <= last_shave; ++shave_i)
    {
        const ShaveResource &sh = res[shave_i];
        resMgr->setupShaveForKernel(sh);
        t_HCWPostOps3DParams *postOpsParams = resMgr->getParams<t_HCWPostOps3DParams>(sh);

        *postOpsParams = *static_cast<const t_HCWPostOps3DParams*>(ddrParams);

        postOpsParams->input        = (input_data  + in_out_offset);
        postOpsParams->output       = (output_data + in_out_offset);
        postOpsParams->offset       = in_out_offset;
        postOpsParams->start_line   = start_line;
        postOpsParams->weights      = weights_data;
        postOpsParams->bias         = biases_data;
        postOpsParams->height       = h_per_shave;

        // Distribute one line of width to the first h_per_shave_remainder shave.
        in_out_offset += h_per_shave * ddrParams->out_step;

        if(h_per_shave_remainder != 0)
        {
            postOpsParams->height += 1;
            in_out_offset += ddrParams->out_step;
            --h_per_shave_remainder;
        }
        start_line += postOpsParams->height;

        resMgr->updateLayerParams(sh, postOpsParams);

#if defined(__leon_rt__) || defined(__leon__)
        rtems_cache_flush_multiple_data_lines(postOpsParams, sizeof(t_HCWPostOps3DParams));
#endif
    }
}

void prePostOpsND(const LayerParams *layerParams, ShaveResourceManager *resMgr) {
    const int VECTOR_SIZE = 8; // Changes to this should be reflected in the code as well

    const PostOpsNDParams* ddrParams = static_cast<const PostOpsNDParams *>(layerParams);

    auto inputData = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(0));

    const half* weightsData = nullptr;
    const half* biasesData = nullptr;

    int input_idx = 1;
    if (ddrParams->hasWeights) {
        weightsData = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        ++input_idx;
    }
    if (ddrParams->hasBiases) {
        biasesData = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        ++input_idx;
    }

    auto outputData = reinterpret_cast<half*>(resMgr->getAbsoluteOutputAddr(0));

    unsigned int numShaves = resMgr->getMaximumShaves();
    auto res = resMgr->requestShaves(numShaves);
    int totalShaves = numShaves; // save requested SHAVEs # to get rid of extra SHAVEs

    // get local copy for geometry transformations
    s32 dims[MAX_ND_DIMS] = {};
    s32 istrides[MAX_ND_DIMS] = {};
    s32 ostrides[MAX_ND_DIMS] = {};

    int ndims = ddrParams->ndims;
    for (int i = 0; i < ndims; ++i) {
        dims[i]     = ddrParams->dims[i];
        istrides[i] = ddrParams->istrides[i];
        ostrides[i] = ddrParams->ostrides[i];
    }

    int axis = ddrParams->axis;

    const bool hasAxis = bool(axis >= 0);
    const bool hasAxisData = (weightsData || biasesData) ? hasAxis : false;
    const int axisDim = hasAxis ? dims[axis] : 0;

    int firstShave = 0;
    int lastShave = firstShave + numShaves - 1;

    const int bpp = sizeof(half); // INPUT_BPP

    // excluding dim == 1 from dims
    int last = ((dims[0] == 1) && ((istrides[1] != bpp) || (ostrides[1] != bpp))) ? 1 : 0;
    for (int i = ndims - 1; i >= last; --i)
    {
        if (ndims <= 1) break;
        if ((dims[i] == 1) && (i != axis))
        {
            // dimension exclusion; strides from outer dim
            ndims = arraysElementExclude(dims, istrides, ostrides, i, ndims);
            axis = (axis >= i) ? (axis - 1) : axis;
        }
        last = ((dims[0] == 1) && ((istrides[1] != bpp) || (ostrides[1] != bpp))) ? 1 : 0;
    }

    // save number of dims for use in 1D tensor split enabling logic
    const int savedNdims = ndims;

    const u32 cmxAvail = SHAVE_LIB_DATA_SIZE;

    // merging dim & dim+1, if it possible, except for axis
    for (int i = 1; i < ndims; ++i)
    {
        const int j = i - 1;
        if ((j != axis) && (i != axis) &&
            (dims[j] * istrides[j] == istrides[i]) &&
            (dims[j] * ostrides[j] == ostrides[i]))
        {
            bool allowed = true;

            if (j == 0) // check if line buffers are fit in memory
            {
                int lineSize = dims[0] * dims[1];
                lineSize = hasAxisData ? nn::math::round_up<VECTOR_SIZE>(lineSize) : lineSize;
#if defined(ASYNC_PIPELINE) // async mode
                int numLines = getTotal(dims + 1, ndims - 1);
                int numBuffers = (numLines > 3) ? 3 : 1;
#else // ASYNC_PIPELINE // sync mode
                int numBuffers = 1;
#endif // ASYNC_PIPELINE
                u32 cmxNeed = numBuffers * lineSize * bpp; //INPUT_BPP;
                allowed = bool(cmxNeed <= cmxAvail);
            }

            if (allowed)
            {
                dims[j] *= dims[i];
                // dimension exclusion; strides from inner dim
                ndims = arraysElementExclude(dims, istrides, ostrides, i, ndims);
                axis = (axis >= i) ? (axis - 1) : axis;
                --i; // next try at the same dim
            }
        }
    }

    // select which dma strategy to be used - 2D vs 3D
    // Mish avoid this since it should use as many shaves as possiple to handle it.
    bool useDma3D = false;
    if (ndims >= 2)
    {
        int lineSize = hasAxisData ? nn::math::round_up<VECTOR_SIZE>(dims[0]) : dims[0];
        int planeSize = lineSize * dims[1];
#if defined(ASYNC_PIPELINE) // async mode
        int numPlanes = getTotal(dims + 2, ndims - 2);
        int numBuffers = (numPlanes > 3) ? 3 : 1;
#else // ASYNC_PIPELINE // sync mode
        int numBuffers = 1;
#endif // ASYNC_PIPELINE
        u32 cmxNeed = numBuffers * planeSize * bpp; //INPUT_BPP;
        useDma3D = bool(cmxNeed <= cmxAvail);
    }

    const int innerDims = useDma3D ? 2 : 1; // # of inner dimensions to be excluded
    const int minOuterDims = 1; // min # of outer dimensions to preserve
    const int minDims = minOuterDims + innerDims;
    if (ndims < minDims)
    {
        for (int i = ndims; i < minDims; ++i)
        {
            istrides[i] = istrides[ndims - 1];
            ostrides[i] = ostrides[ndims - 1];
            dims[i]  = 1;
        }
        ndims = minDims;
    }

    int piecesTotal = getTotal(dims + innerDims, ndims - innerDims);

    int splitAxis = -1;
    if ((numShaves > 1) && (piecesTotal <= 1) && (savedNdims == 1))
    {
        const int tensorSizeThreshold = 8 * 1024; // heuristic: threshold prevents multi-SHAVE splitting of too small tensors
        const int tensorSize = dims[0];

        if (tensorSize >= tensorSizeThreshold)
        {
            if (dims[0] > 1)
            {
                piecesTotal = dims[0];
                splitAxis = 0; // 0'st axis
            }
        }
    }

    const int dim0 = dims[0];
    const int dim1 = useDma3D ? dims[1] : 0;
    const int inStride1 = useDma3D ? istrides[1] : 0;
    const int outStride1 = useDma3D ? ostrides[1] : 0;

    // axis "granularity" is a size of subtensor under axis in terms of lines
    int axisGran = 1;
    if (axis >= 0)
    {
        for (int i = 1; i < axis; ++i)
            axisGran *= dims[i];
    }

    // exclude inner dims
    for (int i = 0; i < innerDims; ++i)
    {
        ndims = arraysElementExclude(dims, istrides, ostrides, 0, ndims);
    }

    numShaves = std::min((int)numShaves, piecesTotal);
    lastShave = (firstShave + numShaves - 1);
    int piecesPerShave = piecesTotal / numShaves;
    int piecesRemain = piecesTotal % numShaves;

    int start = 0;

    int shave_i = firstShave;
    for(; shave_i <= lastShave; ++shave_i)
    {
        int remain = (piecesRemain > 0) ? 1 : 0;
        piecesRemain -= remain;
        int toProcess = piecesPerShave + remain;

        const ShaveResource &sh = res[shave_i];
        resMgr->setupShaveForKernel(sh);
        PostOpsNDParams *postOpsParams = resMgr->getParams<PostOpsNDParams>(sh);

        *postOpsParams = *static_cast<const PostOpsNDParams*>(ddrParams);

        postOpsParams->input      = inputData;
        postOpsParams->output     = outputData;
        postOpsParams->weights    = weightsData;
        postOpsParams->biases     = biasesData;
        postOpsParams->ndims      = ndims;

        for (int i = 0; i < ndims; ++i) postOpsParams->dims[i]     = dims[i];
        for (int i = 0; i < ndims; ++i) postOpsParams->istrides[i] = istrides[i];
        for (int i = 0; i < ndims; ++i) postOpsParams->ostrides[i] = ostrides[i];

        postOpsParams->axisDim    = axisDim;
        postOpsParams->axisGran   = axisGran;
        postOpsParams->axisSize   = axisDim;
        postOpsParams->dim0       = dim0;
        postOpsParams->dim1       = dim1;
        postOpsParams->inStride1  = inStride1;
        postOpsParams->outStride1 = outStride1;
        postOpsParams->start      = start;
        postOpsParams->toProcess  = toProcess;
        postOpsParams->axis       = axis;
        postOpsParams->useDma3D   = useDma3D;

        // fix address/size info if tensor has split
        if (splitAxis >= 0)
        {
            postOpsParams->input    += start;
            postOpsParams->output   += start;

            postOpsParams->dim0      = toProcess;

            if (axis == splitAxis)
            {
                postOpsParams->axisSize = toProcess;
                if (postOpsParams->weights)
                    postOpsParams->weights += start;
                if (postOpsParams->biases)
                    postOpsParams->biases += start;
            }

            postOpsParams->start     = 0;
            postOpsParams->toProcess = 1;
        }

        resMgr->updateLayerParams(sh, postOpsParams);

        postOpsParams->cmxSlice = postOpsParams->cmxData;
        postOpsParams->cmxSize  = postOpsParams->availableCmxBytes;

        start += toProcess;
    }

    if (totalShaves > (int)numShaves) // extra SHAVEs
    {
        int lastShave = (firstShave + totalShaves - 1);
        for(; shave_i <= lastShave; ++shave_i)
        {
            const ShaveResource &sh = res[shave_i];
            resMgr->setupShaveForKernel(sh);
            PostOpsNDParams *postOpsParams = resMgr->getParams<PostOpsNDParams>(sh);

            *postOpsParams = *static_cast<const PostOpsNDParams*>(ddrParams);

            postOpsParams->toProcess = 0;

            resMgr->updateLayerParams(sh, postOpsParams);
        }
    }
}

} // namespace shave_lib
} // namespace nn
