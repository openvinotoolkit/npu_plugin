// {% copyright %}

#include <stdio.h>

#include <sw_layer.h>

#include <sw_shave_res_manager.h>
#include <nn_log.h>
#include <mvSubspaces.h>
#include <param_softmax.h>

#include <mv_types.h>
#include <math.h>
#include <moviVectorTypes.h>
#include <moviVectorConvert.h>

#include <svuCommonShave.h>

#ifdef CONFIG_TARGET_SOC_3720
#include <dma_shave_params_nn.h>
#else
#include <dma_shave_params.h>
#endif

#include "param_mvn.h"

using namespace nn;
using namespace shave_lib;
using namespace sw_params;

namespace {

using namespace subspace;

#define WORK_BUFFER_SIZE (((p->availableCmxBytes)/4))

struct t_MvMVNParamNClasses
{
    half* input;
    Location inLocation;
    half* output;
    Location outLocation;
    u8* cmxslice;
    int32_t availableCmxBytes;
//    u32 grpLeaderShave;
//    u32 grpShavesNo;

    s32 ndims;
    s32 in_dims[MAX_ND_DIMS];
    s32 in_strides[MAX_ND_DIMS];
    s32 out_strides[MAX_ND_DIMS];

    NDOrder storageOrder;
    float intermediate_mean[MAX_ND_DIMS]; // allocate custom size memory

    s32 start;
    s32 toProcess;
    bool inputInCmx;
    bool outputInCmx;
//    s32 this_shave;

    //layer speccific arguments
    uint32_t acrossChannels;
    uint32_t normalize;
    float eps;
};

static void calc_mean_CHW_fp16(const half *line, int W, float* intermedia_mean, int index){
    for(int w = 0; w < W; w++){
        intermedia_mean[index] += *((half *)(line + w));
    }
}

void mvMVN_1(t_MvMVNParamNClasses *p){
    int order = p->storageOrder;

    // assume order CHW, ToDO: change
    int W = p->in_dims[0];
    int H = p->in_dims[1];
    int C = p->in_dims[2];
    int stride = p->in_strides[1] / sizeof(half); // stride = W

    printf("DEBUG_SHAVE (c, h, w): %d %d %d\n", C, H, W);
    printf("DEBUG_SHAVE order: %X\n", order);

    uint32_t normalize_variance = p->normalize;
    int nShaves = 1;
    int idy = 0; // job_num

    half* input  = p->input;
    half* output = p->output;

    s32* dims = p->in_dims;
    s32* istrides = p->in_strides;
    s32* ostrides = p->out_strides;
    s32 ndims = p->ndims;

    printf("DEBUG_SHAVE stride0: %ld\n", istrides[0]);
    printf("DEBUG_SHAVE stride1: %ld\n", istrides[1]);
    printf("DEBUG_SHAVE stride2: %ld\n", istrides[2]);

    half* p_input0  = (p->inputInCmx) ? input : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* p_output0 = (p->outputInCmx) ? output : reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);

    float *intermedia_mean = p->intermediate_mean;

    for (int c = 0; c < C; ++c) {
        intermedia_mean[c * nShaves + idy] = 0;
        // intermedia_mean[buf_size + c * nShaves + idy] = 0;
    }

    if(order == ND_HWC || order == ND_NHWC){

    } else {
        for(int c = 0; c < C; c++){
            int index = c * nShaves + idy;

            for(int h = 0; h < H; h++){
                half* line = input + c * H * stride + h * stride;

                if (normalize_variance) {
                    // unsuported yet
                } else {
                    calc_mean_CHW_fp16(line, W, intermedia_mean, index);
                }
            }
        }
    }
}

void mvMVN_23(t_MvMVNParamNClasses *p){
    int order = p->storageOrder;

    // assume order is CHW, ToDO: change
    int C = p->in_dims[2];
    int H = p->in_dims[1];
    int W = p->in_dims[0];
    int stride = p->in_strides[1] / sizeof(half); // stride = W

    printf("DEBUG_SHAVE (c, h, w): %d %d %d\n", C, H, W);
    printf("DEBUG_SHAVE order: %X\n", order);

    uint32_t normalize_variance = p->normalize;
    uint32_t acrossChannels = p->acrossChannels;
    int nShaves = 1;
    int idy = 0; // job_num

    half* input  = p->input;
    half* output = p->output;

    s32* dims = p->in_dims;
    s32* istrides = p->in_strides;
    s32* ostrides = p->out_strides;
    s32 ndims = p->ndims;

    printf("DEBUG_SHAVE stride0: %ld\n", istrides[0]);
    printf("DEBUG_SHAVE stride1: %ld\n", istrides[1]);
    printf("DEBUG_SHAVE stride2: %ld\n", istrides[2]);

    half* p_input0  = (p->inputInCmx) ? input : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* p_output0 = (p->outputInCmx) ? output : reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);

    float *intermedia_mean = p->intermediate_mean;

    for(int c = 0; c < C; c++){
        float m_acc = 0.f;

        m_acc = intermedia_mean[c];
        m_acc = m_acc / (H * W);

        for(int h = 0; h < H; h++){
            for(int w = 0; w < W; w++){
                int offset = c * H * stride + h * stride + w;
                output[offset] = input[offset] - m_acc;
            }
        }
    }
}

}  // namespace


using namespace subspace;

namespace nn {
namespace shave_lib {

extern "C" {
void mvn(uint32_t lParams) {
    const MvnParams * layerParams = reinterpret_cast<const MvnParams *>(lParams);

    uint8_t * cmxData = nullptr;   // TODO: Restore the possibility of working with DDR tensors 
    int32_t availableCmxBytes = 0;
    // Special DMA to copy layer params from physical DDR
    half* p_act_data = (half*)(layerParams->input.dataAddr); // 0x1F000000
    half* p_act_out = (half*)(layerParams->output.dataAddr); // 0x1F004000
    t_MvMVNParamNClasses mvnParamsCMX;
    t_MvMVNParamNClasses* sp = &mvnParamsCMX;

    sp->storageOrder = layerParams->input.dimsOrder;

    // sp->axis = layerParams->axis;  // axis in arguments in memory notation because tensors are represented as TensorRefNDData

    sp->inputInCmx = true;
    sp->outputInCmx = true;
    sp->ndims = layerParams->input.numDims;

    int32_t *pDims     = (int32_t *)(layerParams->input.dimsAddr);
    int64_t *iPStrides = (int64_t *)(layerParams->input.stridesAddr);
    int64_t *oPStrides = (int64_t *)(layerParams->output.stridesAddr);

    sp->acrossChannels = layerParams->acrossChannels;
    sp->normalize = layerParams->normalize;
    sp->eps = layerParams->eps;

    p_act_out[15 + 0] = iPStrides[0];
    p_act_out[15 + 1] = iPStrides[1];
    p_act_out[15 + 2] = iPStrides[2];
    p_act_out[15 + 3] = iPStrides[3];
    for (int i = 0; i < layerParams->input.numDims; i++) {
        sp->in_dims[i] = pDims[i];
        sp->in_strides[i] = iPStrides[i] / CHAR_BIT;
        sp->out_strides[i] = oPStrides[i] / CHAR_BIT;
    }

    // sp->axisDim = sp->in_dims[sp->axis];
    // if (sp->axis) {
    //     sp->axisIStride = sp->in_strides[sp->axis];
    //     sp->axisOStride = sp->out_strides[sp->axis];
    // } else {
    //     sp->axisIStride = sp->in_strides[1];
    //     sp->axisOStride = sp->out_strides[1];
    // }

    // arrayElementExclude(sp->in_dims, sp->axis, sp->ndims);
    // sp->ndims = arraysElementExclude(sp->in_strides, sp->out_strides, sp->axis, sp->ndims);

    const auto *lp = &mvnParamsCMX;

    int to_process = getTotal(lp->in_dims, lp->ndims);
    unsigned int shaves_no = 1;
    int32_t firstShave = 0;
    int32_t lastShave = firstShave + static_cast<int>(shaves_no) - 1;
    nnLog(MVLOG_DEBUG, "singleShaveMVN: run on %d SHAVEs\n", shaves_no);


    nnLog(MVLOG_DEBUG, "MVNParamNClasses %d\n", __LINE__);
    // one or many softmax sets on one shave
    int step_size = to_process / shaves_no;
    int step_size_rem = to_process % shaves_no;

    int i = firstShave;
    int processed = 0;

    // Use only one shave for now

    t_MvMVNParamNClasses *mvnParamNClasses = &mvnParamsCMX;;
    int to_process_on_shave = step_size + ((step_size_rem-- > 0) ? 1 : 0);
    nnLog(MVLOG_DEBUG, "i %d, to_process_on_shave %d lines, started from %d\n", i, to_process_on_shave, processed);

    mvnParamNClasses->input = reinterpret_cast<half *>(p_act_data);
    mvnParamNClasses->inLocation = sw_params::Location::NN_CMX;//layerParams->input.location;
    mvnParamNClasses->inputInCmx = true;//layerParams->input.location;
    mvnParamNClasses->output = reinterpret_cast<half *>(p_act_out);
    mvnParamNClasses->outLocation = sw_params::Location::NN_CMX;//layerParams->output.location;
    mvnParamNClasses->outputInCmx = true;//layerParams->input.location;

    mvnParamNClasses->cmxslice = cmxData;
    mvnParamNClasses->availableCmxBytes = availableCmxBytes;
    mvnParamNClasses->ndims = lp->ndims;
    for (int32_t i = 0; i < MAX_ND_DIMS; i++) {
        mvnParamNClasses->in_dims[i] = lp->in_dims[i];
        mvnParamNClasses->in_strides[i] = lp->in_strides[i];
        mvnParamNClasses->out_strides[i] = lp->out_strides[i];
    }
    // mvnParamNClasses->axis = lp->axis;
    // mvnParamNClasses->axisDim = lp->axisDim;
    // mvnParamNClasses->axisIStride = lp->axisIStride;
    // mvnParamNClasses->axisOStride = lp->axisOStride;
    mvnParamNClasses->start = processed;
    mvnParamNClasses->toProcess = to_process_on_shave;

    mvMVN_1(mvnParamNClasses);
    mvMVN_23(mvnParamNClasses);

    processed += to_process_on_shave;
}
}

} // namespace shave_lib
} // namespace nn
