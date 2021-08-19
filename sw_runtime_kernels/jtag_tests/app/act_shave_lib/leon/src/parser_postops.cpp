// {% copyright %}
#include "layers/parser_postops.h"
#include "layers/param_postops.h"
#include "layers/pre_postops.h"

//#include "layers/svuSLKernels_EP.h"

#include <nn_cache.h>
#include <nn_log.h>
#include <layer_parser.h>
#include <sw_nn_runtime_types_3600.h>

extern void*  (shvNN0_chw_postOps_3D_core);
extern void*  (shvNN0_hwc_postOps_3D_core);
extern void*  (shvNN0_hcw_postOps_3D_core);

extern void*  (shvNN0_prePostOpsHWC);


namespace nn {
namespace shave_lib {

void layerCleanupPostops(LayerParams *layerParams)
{
    if (layerParams != nullptr)
    {
        //
        // Q: Why wouldn't we define virtual destructor for PostOpsParams?
        // A: We intend to explicitly balance two things in this C++ file:
        //    - PostOpsParams->params allocated as plain-C buffer
        //    - PostOpsParams->params allocated as cache-aligned
        //    So we free this buffer with cache-aligned-free function
        //
        // Q: Why wouldn't we use dynamic_cast on layerParams?
        // A: Because dynamic cast is prohibited for shave_lib (-fno-rtti)
        //    So we believe this layerParams is instance of PostOpsParams
        //
        auto postOpsParams = static_cast<const PostOpsParams *>(layerParams);
        nn::memory::cache_aligned_free(postOpsParams->params);
    }
}


typedef bool PostOpsFunc(TensorRef* input, TensorRef* output, const bool weights, const bool biases,
                     void *params, t_PostOps postOpType, Layer *layer);

bool postOps_3D(TensorRef* input, TensorRef* output, const bool weights, const bool biases,
                       void *params, t_PostOps postOpType, Layer *layer)
{

    NDOrder storageOrder = input->ndOrder;
    if (((storageOrder == ND_NHCW) ||
         (storageOrder == ND_NCHW) ||
         (storageOrder == ND_NHWC)) &&
         input->dimN() != 1) {
            nnLog(MVLOG_ERROR, "Only 3D layout is supported for the kernel");
            return false;
        }

    auto minorDim = input->dims[0];
    auto middleDim = input->dims[1];
    auto majorDim = input->dims[2];

    if (!minorDim || !middleDim || !majorDim) {
        return false;
    }

    auto inputMinorStride = input->strides[0];
    auto inputMiddleStride = input->strides[1];
    auto inputMajorStride = input->strides[2];

    auto outputMinorStride = output->strides[0];
    auto outputMiddleStride = output->strides[1];
    auto outputMajorStride = output->strides[2];

    if (inputMinorStride != sizeof(fp16) || outputMinorStride != sizeof(fp16)) {
        nnLog(MVLOG_ERROR, "Only fp16 precision is supported");
        return false;
    }

    if ((storageOrder == ND_HWC || storageOrder == ND_NHWC) &&
            minorDim == 1 && !weights && !biases &&
            inputMinorStride == sizeof(fp16) && outputMinorStride == sizeof(fp16) &&
            inputMiddleStride == sizeof(fp16) && outputMiddleStride == sizeof(fp16)) {
        storageOrder = ND_NCHW;
        minorDim = middleDim;
        middleDim = majorDim;
        majorDim = 1;

        inputMinorStride = inputMiddleStride;
        inputMiddleStride = inputMajorStride;
        inputMajorStride = inputMiddleStride * middleDim;

        outputMinorStride = outputMiddleStride;
        outputMiddleStride = outputMajorStride;
        outputMajorStride = outputMiddleStride * middleDim;
    }

    // TODO : can we fix that limitation?
    if (inputMajorStride != inputMiddleStride * middleDim ||
        outputMajorStride != outputMiddleStride * middleDim) {
        nnLog(MVLOG_ERROR, "Only minor dim can be aligned");
        return false;
    }

    bool useCHW = false;
    bool useHCW = false;

    // This operation has separate HCW variant,
    if (postOpType == RELU)
    {
        if (storageOrder != ND_HWC && storageOrder != ND_CHW && storageOrder != ND_HCW &&
            storageOrder != ND_NHWC && storageOrder != ND_NCHW && storageOrder != ND_NHCW) {
            return false;
        }

        if (storageOrder == ND_NHCW || storageOrder == ND_HCW)
        {
            useHCW = true;
        }
    }

    // These operations have separate HCW and CHW variants,
    // other operations are independent from layout
    if (postOpType == PRELU ||
        postOpType == BIAS_RELU ||
        postOpType == BIAS_LEAKY_RELU ||
        postOpType == BIAS ||
        postOpType == SCALE ||
        postOpType == SCALE_SHIFT)
    {
        if (storageOrder == ND_HCW || storageOrder == ND_NHCW)
        {
            useHCW = true;
        }
        else if (storageOrder == ND_CHW || storageOrder == ND_NCHW)
        {
            useCHW = true;
        }
    }

    if (useCHW)
    {
        int width = minorDim;
        int height = middleDim;
        int channels = majorDim;
        int in_step = input->strideH() / sizeof(fp16);
        int out_step = output->strideH() / sizeof(fp16);

        std::unique_ptr<t_CHWPostOps3DParams> postOpsParams (new (std::nothrow) t_CHWPostOps3DParams());
        postOpsParams->order        = storageOrder;
        postOpsParams->height       = height;
        postOpsParams->width        = width;
        postOpsParams->postOpType   = postOpType;
        postOpsParams->params       = params;
        postOpsParams->in_step      = in_step;
        postOpsParams->out_step     = out_step;
        postOpsParams->channels     = channels;

        postOpsParams->has_weights = weights;
        postOpsParams->has_biases = biases;

        cache::flush(postOpsParams.get(), sizeof(t_CHWPostOps3DParams));

        unsigned int id = MVCNN::SoftwareLayerParams::SoftwareLayerParams_PostOpsParams;
        layer->setParams(id, static_cast<LayerParams *>(postOpsParams.release()));

//        layer->setPreamble(reinterpret_cast<void (*)(void*)>(&shvNN0_prePostOpsCHW);
        //layer->setKernelEntry(KERNEL_FUNC(chw_postOps_3D_core));
        layer->setKernelEntry(reinterpret_cast<void (*)(void*)>(&shvNN0_chw_postOps_3D_core));
//        layer->setLayerCleanup(&layerCleanupPostops);
    }
    else if (useHCW)
    {
        int width = minorDim;
        int height = middleDim * majorDim;
        int channels = middleDim;
        int lines = majorDim;
        int in_stride = input->strideC() / sizeof(half);
        int out_stride = output->strideC() / sizeof(half);

        // Nothing to process.
        if(width == 0)
            return false;

        std::unique_ptr<t_HCWPostOps3DParams> postOpsParams (new (std::nothrow) t_HCWPostOps3DParams());
        postOpsParams->order        = storageOrder;
        postOpsParams->width        = width;
        postOpsParams->height       = height;
        postOpsParams->in_step      = in_stride;
        postOpsParams->out_step     = out_stride;
        postOpsParams->postOpType   = postOpType;
        postOpsParams->params       = params;
        postOpsParams->lines        = lines;
        postOpsParams->channels     = channels;

        postOpsParams->has_weights = weights;
        postOpsParams->has_biases = biases;

        cache::flush(postOpsParams.get(), sizeof(t_HCWPostOps3DParams));

        unsigned int id = MVCNN::SoftwareLayerParams::SoftwareLayerParams_PostOpsParams;
        layer->setParams(id, static_cast<LayerParams *>(postOpsParams.release()));

//        layer->setPreamble(PREAMBLE_FUNC(prePostOpsHCW));
        //layer->setKernelEntry(KERNEL_FUNC(hcw_postOps_3D_core));
        layer->setKernelEntry(reinterpret_cast<void (*)(void*)>(&shvNN0_hcw_postOps_3D_core));
//        layer->setLayerCleanup(&layerCleanupPostops);
    }
    else // HWC
    {
        int width = minorDim;
        int height = middleDim * majorDim;
        int in_stride = inputMiddleStride / sizeof(half);
        int out_stride = outputMiddleStride / sizeof(half);

        std::unique_ptr<t_HWCPostOps3DParams> postOpsParams (new (std::nothrow) t_HWCPostOps3DParams());

        postOpsParams->order = storageOrder;
        postOpsParams->width        = width;
        postOpsParams->height        = height;
        postOpsParams->in_step    = in_stride;
        postOpsParams->out_step   = out_stride;
        postOpsParams->postOpType   = postOpType;
        postOpsParams->params       = params;

        postOpsParams->has_weights = weights;
        postOpsParams->has_biases = biases;
        cache::flush(postOpsParams.get(), sizeof(t_HWCPostOps3DParams));

        unsigned int id = MVCNN::SoftwareLayerParams::SoftwareLayerParams_PostOpsParams;
        layer->setParams(id, static_cast<LayerParams *>(postOpsParams.release()));
        layer->setPreamble(reinterpret_cast<preamble>(&shvNN0_prePostOpsHWC));
        layer->setKernelEntry(reinterpret_cast<void (*)(void*)>(&shvNN0_hwc_postOps_3D_core));
//        printf("pre_ptr = %p\n", &shvNN0_prePostOpsHWC);
//        printf("pre_ptr2 = %p\n", layer->pre);
        //layer->setKernelEntry(KERNEL_FUNC(hwc_postOps_3D_core));
//        layer->setLayerCleanup(&layerCleanupPostops);
    }

    return true;
}

bool PostOpsParser::parse(const MVCNN::UPALayerTask *task, Layer *layer) {
    if (task->softLayerParams_type() == MVCNN::SoftwareLayerParams::SoftwareLayerParams_PostOpsParams) {
        if (!parseUPATensors(task, layer)) {
            nnLog(MVLOG_ERROR, "PostOps : input/output tensors parsing fails.");
            return false;
        }

        const MVCNN::PostOpsParams *gfParams = task->softLayerParams_as_PostOpsParams();

        PostOpsFunc* postOps = postOps_3D;

        auto inRefs = layer->getInputs();
        auto outRefs = layer->getOutputs();
        auto* inref = &inRefs[0];
        auto* outref = &outRefs[0];

        if (inref->ndOrder != outref->ndOrder) {
            nnLog(MVLOG_ERROR, "Input and Output order mismatch");
            return false;
        }

        bool success = true;

        switch (gfParams->nested_params_type())
        {
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_ClampParams: {
            auto clamp_params = gfParams->nested_params_as_ClampParams();
            auto cp = reinterpret_cast<t_ClampLayerParams *>(
                nn::memory::cache_aligned_alloc(sizeof(t_ClampLayerParams)));
            if (cp == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for Clamp");
                success = false;
                break;
            }
            cp->min = clamp_params->min();
            cp->max = clamp_params->max();
            cache::flush(cp, sizeof(t_ClampLayerParams));
            success = postOps(inref, outref, false, false, cp, CLAMP, layer);
            if (!success) {
                nn::memory::cache_aligned_free(cp);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_EluParams: {
            auto params = gfParams->nested_params_as_EluParams();
            auto x = reinterpret_cast<float *>(nn::memory::cache_aligned_alloc(sizeof(float)));
            if (x == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for ELU");
                success = false;
                break;
            }
            *x = params->x();
            cache::flush(x, sizeof(float));
            success = postOps(inref, outref, false, false, x, ELU, layer);
            if (!success) {
                nn::memory::cache_aligned_free(x);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_BiasLeakyReluParams: {
            auto params = gfParams->nested_params_as_BiasLeakyReluParams();
            auto x = reinterpret_cast<float *>(nn::memory::cache_aligned_alloc(sizeof(float)));
            if (x == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for BiasLeakyReLU");
                success = false;
                break;
            }
            *x = params->negative_slope();
            cache::flush(x, sizeof(float));
            success = postOps(inref, outref, false, true, x, BIAS_LEAKY_RELU, layer);
            if (!success) {
                nn::memory::cache_aligned_free(x);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_BiasReluParams: {
            auto params = gfParams->nested_params_as_BiasReluParams();
            auto x = reinterpret_cast<float *>(nn::memory::cache_aligned_alloc(sizeof(float)));
            if (x == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for BiasReLU");
                success = false;
                break;
            }
            *x = params->negative_slope();
            cache::flush(x, sizeof(float));
            success = postOps(inref, outref, false, true, x, BIAS_RELU, layer);
            if (!success) {
                nn::memory::cache_aligned_free(x);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_LeakyReluParams: {
            auto params = gfParams->nested_params_as_LeakyReluParams();
            auto x = reinterpret_cast<float *>(nn::memory::cache_aligned_alloc(sizeof(float)));
            if (x == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for LeakyReLU");
                success = false;
                break;
            }
            *x = params->negative_slope();
            cache::flush(x, sizeof(float));
            success = postOps(inref, outref, false, false, x, LEAKY_RELU, layer);
            if (!success) {
                nn::memory::cache_aligned_free(x);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_ReluParams: {
            //auto params = gfParams->nested_params_as_ReluParams();
            // TODO: add parameter to the schema to support clamp max in ReLU (ex. ReLU6)
            auto x = reinterpret_cast<float *>(nn::memory::cache_aligned_alloc(sizeof(float)));
            if (x == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for ReLU");
                success = false;
                break;
            }
            *x = 0.0f;
            cache::flush(x, sizeof(float));
            success = postOps(inref, outref, false, false, x, RELU, layer);
            if (!success) {
                nn::memory::cache_aligned_free(x);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_PReluParams: {
            success = postOps(inref, outref, true, false, nullptr, PRELU, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_PowerParams: {
            auto params = gfParams->nested_params_as_PowerParams();
            auto pp = reinterpret_cast<t_PowerLayerParams *>(
                nn::memory::cache_aligned_alloc(sizeof(t_PowerLayerParams)));
            if (pp == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for Power");
                success = false;
                break;
            }
            pp->shift = params->shift();
            pp->scale = params->scale();
            pp->power = params->power();
            cache::flush(pp, sizeof(t_PowerLayerParams));
            success = postOps(inref, outref, false, false, pp, POWER, layer);
            if (!success) {
                nn::memory::cache_aligned_free(pp);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_SigmoidParams: {
            success = postOps(inref, outref, false, false, nullptr, SIGMOID, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_TanhParams: {
            success = postOps(inref, outref, false, false, nullptr, TANH, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_BiasParams: {
            success = postOps(inref, outref, false, true, nullptr, BIAS, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_ScaleParams: {
            success = postOps(inref, outref, true, false, nullptr, SCALE, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_ScaleShiftParams: {
            success = postOps(inref, outref, true, true, nullptr, SCALE_SHIFT, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_HSwishParams: {
            success = postOps(inref, outref, false, false, nullptr, HSWISH, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_SwishParams: {
            auto params = gfParams->nested_params_as_SwishParams();
            auto pp = reinterpret_cast<t_SwishLayerParams *>(
                nn::memory::cache_aligned_alloc(sizeof(t_SwishLayerParams)));
            if (pp == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for Swish");
                success = false;
                break;
            }
            pp->beta = params->beta();
            cache::flush(pp, sizeof(t_SwishLayerParams));
            success = postOps(inref, outref, false, false, pp, SWISH, layer);
            if (!success) {
                nn::memory::cache_aligned_free(pp);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_SoftPlusParams: {
            success = postOps(inref, outref, false, false, nullptr, SOFTPLUS, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_MishParams: {
            success = postOps(inref, outref, false, false, nullptr, MISH, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_FloorParams: {
            success = postOps(inref, outref, false, false, nullptr, FLOOR, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_RoundParams: {
            auto params = gfParams->nested_params_as_RoundParams();
            auto pp = reinterpret_cast<t_RoundLayerParams *>(
                nn::memory::cache_aligned_alloc(sizeof(t_RoundLayerParams)));
            if (pp == nullptr) {
                mvLog(MVLOG_ERROR, "PostOpsParser::parse cannot allocate params for Round");
                success = false;
                break;
            }
            pp->mode = static_cast<roundMode>(params->mode()); // enum-to-enum: same int value
            cache::flush(pp, sizeof(t_RoundLayerParams));
            success = postOps(inref, outref, false, false, pp, ROUND, layer);
            if (!success) {
                nn::memory::cache_aligned_free(pp);
            }
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_CeilingParams: {
            success = postOps(inref, outref, false, false, nullptr, CEIL, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_ErfParams: {
            success = postOps(inref, outref, false, false, nullptr, ERF, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_GeluParams: {
            success = postOps(inref, outref, false, false, nullptr, GELU, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_LogParams: {
            success = postOps(inref, outref, false, false, nullptr, LOG, layer);
            break;
        }
        case MVCNN::PostOpsNestedParams::PostOpsNestedParams_ExpParams: {
            success = postOps(inref, outref, false, false, nullptr, EXP, layer);
            break;
        }
        default: {
            success = false;
            nnLog(MVLOG_ERROR, "Unsupported postop type");
            break;
        }
        }

        return success;
    }

    return false;
}

} // namespace shave_lib
} // namespace nn
