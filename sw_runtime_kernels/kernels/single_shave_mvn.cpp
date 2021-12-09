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

    s32 nShaves;
    s32 jobNum;

    NDOrder storageOrder;
    s32 channels;
    s32 height;
    s32 width;
    s32 stride;

    s32 start;
    s32 toProcess;
    bool inputInCmx;
    bool outputInCmx;
//    s32 this_shave;

    uint32_t acrossChannels;
    uint32_t normalize;
    float eps;

    float intermediate_mean[MAX_ND_DIMS];
};

static void calc_mean_CHW_fp16(const half *line, int W, float* intermedia_mean, int index){
    for(int w = 0; w < W; w++){
        intermedia_mean[index] += *((half *)(line + w));
    }
}

static void calc_mean_var_CHW_fp16(const half *line, int W, float* intermedia_mean, int index, int buf_size){
    for(int w = 0; w < W; w++){
        half temp = *((half *)(line + w));

        intermedia_mean[index] += (float)temp;
        intermedia_mean[buf_size + index] += (float)temp * (float)temp;
    }
}

void mvMVN_1(t_MvMVNParamNClasses *p){
    NDOrder order = p->storageOrder;

    int W = p->width;
    int H = p->height;
    int C = p->channels;
    int stride = p->stride;

    // printf("DEBUG_SHAVE (c, h, w): %d %d %d\n", C, H, W);
    // printf("DEBUG_SHAVE order: %X\n", order);

    uint32_t normalize_variance = p->normalize;
    int nShaves = p->nShaves; // Use only one shave for now
    int idy = p->jobNum;

    int buf_size = nShaves * C;

    half* input  = p->input;
    half* output = p->output;

    // printf("DEBUG_SHAVE stride0: %ld\n", istrides[0]);
    // printf("DEBUG_SHAVE stride1: %ld\n", istrides[1]);
    // printf("DEBUG_SHAVE stride2: %ld\n", istrides[2]);

    half* p_input0  = (p->inputInCmx) ? input : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* p_output0 = (p->outputInCmx) ? output : reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);

    float *intermedia_mean = p->intermediate_mean;

    for (int c = 0; c < C; ++c) {
        intermedia_mean[c * nShaves + idy] = 0;
        intermedia_mean[buf_size + c * nShaves + idy] = 0;
    }

    for(int c = 0; c < C; c++){
        int index = c * nShaves + idy;

        for(int h = 0; h < H; h++){
            half* line = input + c * H * stride + h * stride;

            if (!normalize_variance) {
                calc_mean_CHW_fp16(line, W, intermedia_mean, index);
            } else {
                calc_mean_var_CHW_fp16(line, W, intermedia_mean, index, buf_size);
            }
        }
    }
}

void mvMVN_23(t_MvMVNParamNClasses *p){
    NDOrder order = p->storageOrder;

    int W = p->width;
    int H = p->height;
    int C = p->channels;
    int stride = p->stride;

    // printf("DEBUG_SHAVE (c, h, w): %d %d %d\n", C, H, W);
    // printf("DEBUG_SHAVE order: %X\n", order);

    uint32_t normalize_variance = p->normalize;
    uint32_t acrossChannels = p->acrossChannels;
    float epsilon = p->eps;
    int nShaves = p->nShaves; // Use only one shave for now
    int idy = p->jobNum;

    const float* variance_part = p->intermediate_mean + nShaves * C;
    const float* mean_part = p->intermediate_mean;

    half* input  = p->input;
    half* output = p->output;

    // printf("DEBUG_SHAVE stride0: %ld\n", istrides[0]);
    // printf("DEBUG_SHAVE stride1: %ld\n", istrides[1]);
    // printf("DEBUG_SHAVE stride2: %ld\n", istrides[2]);

    half* p_input0  = (p->inputInCmx) ? input : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* p_output0 = (p->outputInCmx) ? output : reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);

    float mean;
    float variance;

    for(int c = 0; c < C; c++){
        float m_acc;
        float v_acc;

        m_acc = mean_part[c];
        v_acc = variance_part[c];
        m_acc = m_acc / (H * W);
        v_acc = v_acc / (H * W);
        v_acc = v_acc - m_acc * m_acc;
        v_acc = v_acc < 0 ? 1.f : v_acc;
        v_acc = sqrtf(v_acc) + epsilon;

        mean = m_acc;
        variance = 1.f / v_acc;

        for(int h = 0; h < H; h++){
            for(int w = 0; w < W; w++){
                int offset = c * H * stride + h * stride + w;
                output[offset] = input[offset] - mean;
                if(normalize_variance){
                    output[offset] = output[offset] * variance;
                }
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

    sp->inputInCmx = true;
    sp->outputInCmx = true;

    int32_t *pDims     = (int32_t *)(layerParams->input.dimsAddr);
    int64_t *iPStrides = (int64_t *)(layerParams->input.stridesAddr);
    int64_t *oPStrides = (int64_t *)(layerParams->output.stridesAddr);

    sp->acrossChannels = layerParams->acrossChannels;
    sp->normalize = layerParams->normalize;
    sp->eps = layerParams->eps;

    if(layerParams->input.dimsOrder != ND_CHW && layerParams->input.dimsOrder != ND_NCHW){
        nnLog(MVLOG_ERROR, "Unsuported layout, expected CHW or NCHW");
        return;
    }
    sp->storageOrder = layerParams->input.dimsOrder;

    s32 ndims = layerParams->input.numDims;
    bool success;
    NDDims indices = orderNDToIndices(layerParams->input.dimsOrder, success);
    sp->channels = pDims[indices[ndims - 3]];
    sp->height = pDims[indices[ndims - 2]];
    sp->width = pDims[indices[ndims - 1]];
    sp->stride = iPStrides[indices[ndims - 2]] / (CHAR_BIT * sizeof(half)); // stride = W

    // printf("SHAVE_DEBUG: ndims = %d\n", layerParams->input.numDims);

    const auto *lp = &mvnParamsCMX;

    int to_process = getTotal(pDims, ndims);
    unsigned int shaves_no = 1;
    int32_t firstShave = 0;
    int32_t jobNum = 0;
    int32_t lastShave = firstShave + static_cast<int>(shaves_no) - 1;
    nnLog(MVLOG_DEBUG, "singleShaveMVN: run on %d SHAVEs\n", shaves_no);

    nnLog(MVLOG_DEBUG, "MVNParamNClasses %d\n", __LINE__);
    // one or many softmax sets on one shave
    int step_size = to_process / shaves_no;
    int step_size_rem = to_process % shaves_no;

    int i = firstShave;
    int processed = 0;

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

    mvnParamNClasses->start = processed;
    mvnParamNClasses->toProcess = to_process_on_shave;

    mvnParamNClasses->nShaves = shaves_no;
    mvnParamNClasses->jobNum = jobNum;

    mvMVN_1(mvnParamNClasses);
    mvMVN_23(mvnParamNClasses);

    processed += to_process_on_shave;
}
}

} // namespace shave_lib
} // namespace nn
