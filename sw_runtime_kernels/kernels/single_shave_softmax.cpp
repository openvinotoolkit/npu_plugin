//
// Copyright 2022 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <param_softmax.h>
#include <mvSubspaces.h>
#define INFINITY (1.0f /0.0f)
#include <moviVectorConvert.h>

#ifdef CONFIG_TARGET_SOC_3720
#include <dma_shave_nn.h>
#else
#include <dma_shave.h>
#endif

using namespace sw_params;
using namespace subspace;

namespace {

typedef half* phalf;

#define WORK_BUFFER_SIZE (((p->availableCmxBytes)/4))

#define SWAP(a, b, type)\
{                       \
    type temp = (a);    \
    a = b;              \
    b = temp;           \
}

#define B_8_EXP(_in,_out)                                   \
    (_out)[0] = __builtin_shave_sau_exp2_f16_l_r((_in)[0]); \
    (_out)[1] = __builtin_shave_sau_exp2_f16_l_r((_in)[1]); \
    (_out)[2] = __builtin_shave_sau_exp2_f16_l_r((_in)[2]); \
    (_out)[3] = __builtin_shave_sau_exp2_f16_l_r((_in)[3]); \
    (_out)[4] = __builtin_shave_sau_exp2_f16_l_r((_in)[4]); \
    (_out)[5] = __builtin_shave_sau_exp2_f16_l_r((_in)[5]); \
    (_out)[6] = __builtin_shave_sau_exp2_f16_l_r((_in)[6]); \
    (_out)[7] = __builtin_shave_sau_exp2_f16_l_r((_in)[7]);

static void softmax(fp16 *input, int n, int istride, fp16 *output, int ostride)
{
    if(n == 1)
    {
        output[0] = 1;
        return;
    }
    float sumf = 0.0f;
    fp16 * in;
    fp16 * out;
    const ushort ulog_2_e = 0x3dc5;  // log2(exp(1.0f))
    const fp16 log_2_e = ((fp16*)&ulog_2_e)[0];
    in = input;
    fp16 largest;
    largest = in[0 * istride];
    fp16 r = in[1 * istride];
    for(int i = 1; i < n - 1; ++i)
    {
        fp16 r_ = in[(i + 1) * istride];
        largest = __builtin_shave_cmu_max_f16_rr_half(largest, r);
        r = r_;
    }
    {
        largest = __builtin_shave_cmu_max_f16_rr_half(largest, r);
    }

    out = output;
    r = (in[0 * istride] - largest);
    for(int i = 0; i < n - 1; ++i)
    {
        fp16 r_ = (in[(i + 1) * istride] - largest);
        half f_exp = __builtin_shave_sau_exp2_f16_l_r((r) * log_2_e);
        out[i * ostride] = f_exp;
        sumf += (float)f_exp;
        r = r_;
    }
    {
        half f_exp = __builtin_shave_sau_exp2_f16_l_r((r) * log_2_e);
        out[(n - 1) * ostride] = f_exp;
        sumf += (float)f_exp;
    }
    half sum;
    if(!((half)sumf))
        sum = (half)INFINITY;
    else
        sum = (half)1.f/(half)sumf;
    r = out[0 * ostride];
    for(int i = 0; i < n - 1; ++i)
    {
        half r_ = out[(i + 1) * ostride];
        out[i * ostride] = r * sum;
        r = r_;
    }
    out[(n - 1) * ostride] = r * sum;
}

static void softmax8(fp16 *input, int n, int istride, fp16 *output, int ostride)
{
    if(n == 1)
    {
        ((half8*)(output))[0] = 1;
        return;
    }
    fp16 * in;
    fp16 * out;
    const ushort ulog_2_e = 0x3dc5;  // log2(exp(1.0f))
    const half8 log_2_e = ((fp16*)&ulog_2_e)[0];
    in = input;
    half8 vlargest;
    vlargest = ((half8*)(in + 0 * istride))[0];
    half8 r = ((half8*)(in + 1 * istride))[0];
    for(int i = 1; i < n - 1; ++i)
    {
        half8 r_ = ((half8*)(in + (i + 1) * istride))[0];
        vlargest = __builtin_shave_cmu_max_f16_rr_half8(vlargest, r);
        r = r_;
    }
    {
        vlargest = __builtin_shave_cmu_max_f16_rr_half8(vlargest, r);
    }

    out = output;
    r = (((half8*)(in + 0 * istride))[0] - vlargest) * log_2_e;
    float8 sumf = 0.f;
    for(int i = 0; i < n - 1; ++i)
    {
        half8 r_ = (((half8*)(in + (i + 1) * istride))[0] - vlargest);

        half8 vexp0 = (half8)0; B_8_EXP(r, vexp0);

        ((half8*)(out + i * ostride))[0] = vexp0;
        sumf += mvuConvert_float8(vexp0);
        r = r_ * log_2_e;
    }
    {

        half8 vexp0 = (half8)0; B_8_EXP(r, vexp0);

        ((half8*)(out + (n - 1) * ostride))[0] = vexp0;
        sumf += mvuConvert_float8(vexp0);
    }

    half8 sum = {(half)1.f/(half)sumf[0], (half)1.f/(half)sumf[1], (half)1.f/(half)sumf[2], (half)1.f/(half)sumf[3]
               , (half)1.f/(half)sumf[4], (half)1.f/(half)sumf[5], (half)1.f/(half)sumf[6], (half)1.f/(half)sumf[7]};

    r = ((half8*)(out + (0) * ostride))[0];
    for(int i = 0; i < n - 1; ++i)
    {
        half8 r_ = ((half8*)(out + (i + 1) * ostride))[0];
        ((half8*)(out + i * ostride))[0] = r * sum;
        r = r_;
    }
    {
        ((half8*)(out + (n - 1) * ostride))[0] = r * sum;
    }
}

static fp16 largest_softmax_C(fp16 *input, int n)
{
    int i;
    fp16 largest;
    if(n <= 8)
    {
        largest = input[0];
        for(i = 1; i < n; ++i)
        {
            largest = __builtin_shave_cmu_max_f16_rr_half(largest, input[i]);
        }
    }
    else
    {
        half8 * in;
        in = (half8*)input;
        half8 vlargest = in[0];
        in++;
        for(i = 8; i < n-7; i += 8, in++)
        {
            vlargest = __builtin_shave_cmu_max_f16_rr_half8(vlargest, in[0]);
        }
        vlargest = __builtin_shave_cmu_max_f16_rr_half8(vlargest, ((half8*)(input+n-8))[0]);
        largest = vlargest[0];

        for(i = 1; i < 8; i ++)
        {
            largest = __builtin_shave_cmu_max_f16_rr_half(largest, vlargest[i]);
        }
    }
    return largest;
}

static float sum_softmax_C(fp16 *input, int n, fp16 *output, fp16 largest)
{
    int i;
    float sumf = 0.0f;
    const ushort ulog_2_e = 0x3dc5;  // log2(exp(1.0f))
    const half8 log_2_e = ((fp16*)&ulog_2_e)[0];
    const fp16 slog_2_e = ((fp16*)&ulog_2_e)[0];

    if (n <= 8)
    {
        for(i = 0; i < n; ++i)
        {
            half f_exp = __builtin_shave_sau_exp2_f16_l_r((input[i] - largest) * slog_2_e);
            output[i] = f_exp;
            sumf += (float)f_exp;
        }
    }
    else
    {
        half8 * in8 = (half8*)input;
        half8 * out8 = (half8*)output;
        half8 r = (in8[0] - largest) * log_2_e;
        in8++;

        float8 sumf8 = 0;

        for(i = 0; i < n - 7 - 8; i += 16, in8 += 2, out8 += 2)
        {
            half8 r_ = (in8[0] - largest);
            half8 r__ = (in8[1] - largest);
            {

                half8 vexp0 = (half8)0; B_8_EXP(r, vexp0);
                out8[0] = vexp0;
                sumf8 += mvuConvert_float8(vexp0);
            }
            {
                r = r_ * log_2_e;

                half8 vexp1 = (half8)0; B_8_EXP(r, vexp1);
                out8[1] = vexp1;
                sumf8 += mvuConvert_float8(vexp1);
            }

            r = r__ * log_2_e;
        }
        if(i < n - 7)
        {
            {

                half8 vexp0 = (half8)0; B_8_EXP(r, vexp0);
                out8[0] = vexp0;
                sumf8 += mvuConvert_float8(vexp0);
            }
            i += 8;
        }

        sumf = __builtin_shave_sau_sumx_f32_r(sumf8.lo) + __builtin_shave_sau_sumx_f32_r(sumf8.hi);
        for(; i < n; ++i)
        {
            half f_exp = __builtin_shave_sau_exp2_f16_l_r(((input[i] - largest)) * slog_2_e);
            output[i] = f_exp;
            sumf += (float)f_exp;
        }
    }
    return sumf;
}

static void calculateSoftMaxInner(const int dimX, int n, half* input, half* output,
        int, int, int, int)
{
    for(int w = 0; w < dimX; ++w)
    {
        half* in = input + w * n;
        half* out = output + w * n;
        float sumf = sum_softmax_C(in, n, out, largest_softmax_C(in, n));
        half sum;
        if(!((half)sumf))
            sum = (half)INFINITY;
        else
            sum = (half)1.f/(half)sumf;
        for(int j = 0; j < n; ++j)
        {
            out[j] *= sum;
        }
    }
}

static void calculateSoftMaxOuter(const int dimX, int n, half* input, half* output,
        int istride1, int istride2, int ostride1, int ostride2)
{
    int w = 0;
    for(; w < dimX - 7; w += 8)
    {
        softmax8(input + w * istride1, n, istride2, output + w * ostride1, ostride2);
    }
    for(; w < dimX; ++w)
    {
        softmax(input + w * istride1, n, istride2, output + w * ostride1, ostride2);
    }
}

struct t_MvSoftMaxParamNClasses
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

    s32 axis;
    s32 axisDim;
    s32 axisIStride;
    s32 axisOStride;
    s32 start;
    s32 toProcess;
    bool inputInCmx;
    bool outputInCmx;
//    s32 this_shave;
};

// softmax on inner/outer dimensions
void mvSoftMaxSingle(t_MvSoftMaxParamNClasses *p)
{
    half* in  = p->input;
    half* out = p->output;

    s32* dims = p->in_dims;
    s32* istrides = p->in_strides;
    s32* ostrides = p->out_strides;
    s32 ndims = p->ndims;

    DmaAlShave dmaRTask;

    half* p_input0  = (p->inputInCmx) ? in : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* p_output0 = (p->outputInCmx) ? out : reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);

    int sets_per_step = (WORK_BUFFER_SIZE) / (sizeof(half) * p->axisDim);
    int32_t setCoords[MAX_ND_DIMS];

    void (*calculate)(const int dimX, int n, half* input, half* output,
            int istride1, int istride2, int ostride1, int ostride2);
    calculate = (p->axis > 0) ? calculateSoftMaxOuter : calculateSoftMaxInner;
    {
        sets_per_step = (p->inputInCmx && p->outputInCmx) ? p->toProcess : (2 * WORK_BUFFER_SIZE) / (sizeof(half) * p->axisDim);
        int i = p->start;
        subspace::getCoord(i, dims, ndims, setCoords);
        s32 r_step;
        s32 axisDim = p->axisDim;

        s32* dmaWidth = (p->axis) ? &r_step : &axisDim;
        s32 iStride = p->axisIStride / sizeof(fp16);
        s32 oStride = p->axisOStride / sizeof(fp16);
        s32* ir_stride = (p->inputInCmx) ? &iStride : &r_step;
        s32* or_stride = (p->outputInCmx) ? &oStride : &r_step;
        while(i < p->start + p->toProcess)
        {
            r_step = __builtin_shave_cmu_min_i32_rr_int(sets_per_step, dims[0] - setCoords[0]);
            unsigned inOffset, outOffset;
            subspace::getOffsetsU8(setCoords, istrides, ostrides, ndims, inOffset, outOffset);
            if (p->inputInCmx) {
                p_input0 = (half*)((u8*)in + inOffset);
            } else {
                dmaRTask.start((u8*)in + inOffset, (u8 *)p_input0,
                      sizeof(fp16) * p->axisDim * r_step,
                      sizeof(fp16) * dmaWidth[0],
                      sizeof(fp16) * dmaWidth[0],
                      p->axisIStride,
                      sizeof(fp16) * dmaWidth[0]);
                dmaRTask.wait();
            }
            if (p->outputInCmx) {
                p_output0 = (half*)((u8*)out + outOffset);
            }
            calculate(r_step, p->axisDim, p_input0, p_output0, 1, *ir_stride, 1, *or_stride);
            if (!(p->outputInCmx)) {
                dmaRTask.start((u8 *)p_output0, (u8*)out + outOffset,
                      sizeof(fp16) * p->axisDim * r_step,
                      sizeof(fp16) * dmaWidth[0],
                      sizeof(fp16) * dmaWidth[0],
                      sizeof(fp16) * dmaWidth[0],
                      p->axisOStride);
                dmaRTask.wait();
            }
            i += r_step;
            subspace::incrementNCoord(setCoords, dims, ndims, r_step);
        }
    }
}

}  // namespace

using namespace subspace;

namespace nn {
namespace shave_lib {

extern "C" {

void singleShaveSoftmax(uint32_t lParams) {
    uint8_t * cmxData = nullptr;   // TODO: Restore the possibility of working with DDR tensors
    int32_t availableCmxBytes = 0;
    // Special DMA to copy layer params from physical DDR
    half* p_act_data = (half*)(reinterpret_cast<SoftmaxParams*>(lParams)->input.dataAddr); // 0x1F000000
    half* p_act_out = (half*)(reinterpret_cast<SoftmaxParams*>(lParams)->output.dataAddr); // 0x1F004000
    const SoftmaxParams * layerParams = reinterpret_cast<const SoftmaxParams *>(lParams);
    t_MvSoftMaxParamNClasses softmaxParamsCMX;
    t_MvSoftMaxParamNClasses* sp = &softmaxParamsCMX;

    // parameters specific for softmax in customCpp parameter buffer
    sp->axis = layerParams->axis;  // axis in arguments in memory notation because tensors are represented as TensorRefNDData
                         // which is in memory notation too
    sp->inputInCmx = true;//(layerParams->input.location == sw_params::Location::NN_CMX || layerParams->input.location == sw_params::Location::UPA_CMX);
    sp->outputInCmx = true;//(layerParams->output.location == sw_params::Location::NN_CMX || layerParams->output.location == sw_params::Location::UPA_CMX);
    sp->ndims = layerParams->input.numDims;

    int32_t *pDims     = (int32_t *)(layerParams->input.dimsAddr);
    int64_t *iPStrides = (int64_t *)(layerParams->input.stridesAddr);
    int64_t *oPStrides = (int64_t *)(layerParams->output.stridesAddr);

    p_act_out[15 + 0] = iPStrides[0];
    p_act_out[15 + 1] = iPStrides[1];
    p_act_out[15 + 2] = iPStrides[2];
    p_act_out[15 + 3] = iPStrides[3];
    for (int i = 0; i < layerParams->input.numDims; i++) {
        sp->in_dims[i] = pDims[i];
        sp->in_strides[i] = iPStrides[i] / CHAR_BIT;
        sp->out_strides[i] = oPStrides[i] / CHAR_BIT;
    }

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

//    *** Part of code copied from parser with commented out fragments  END***

    const auto *lp = &softmaxParamsCMX;

    int to_process = getTotal(lp->in_dims, lp->ndims);
    unsigned int shaves_no = 1;
    int32_t firstShave = 0;
    int32_t lastShave = firstShave + static_cast<int>(shaves_no) - 1;
    nnLog(MVLOG_DEBUG, "singleShaveSoftmax: run on %d SHAVEs\n", shaves_no);

    {
        nnLog(MVLOG_DEBUG, "softMaxParamNClasses %d\n", __LINE__);
        // one or many softmax sets on one shave
        int step_size = to_process / shaves_no;
        int step_size_rem = to_process % shaves_no;

        nnLog(MVLOG_DEBUG, "axis %d, step_size %d, to_process %d, shaves_no %d\n", lp->axis,
                    step_size, to_process, shaves_no);

        int i = firstShave;
        int processed = 0;

        for (; i <= lastShave/* && processed < to_process*/; i++) {
            t_MvSoftMaxParamNClasses *softMaxParamNClasses = &softmaxParamsCMX;;
            int to_process_on_shave = step_size + ((step_size_rem-- > 0) ? 1 : 0);
            nnLog(MVLOG_DEBUG, "i %d, to_process_on_shave %d lines, started from %d\n", i, to_process_on_shave, processed);

            softMaxParamNClasses->input = reinterpret_cast<half *>(p_act_data);
            softMaxParamNClasses->inLocation = sw_params::Location::NN_CMX;//layerParams->input.location;
            softMaxParamNClasses->inputInCmx = true;//layerParams->input.location;
            softMaxParamNClasses->output = reinterpret_cast<half *>(p_act_out);
            softMaxParamNClasses->outLocation = sw_params::Location::NN_CMX;//layerParams->output.location;
            softMaxParamNClasses->outputInCmx = true;//layerParams->input.location;

            softMaxParamNClasses->cmxslice = cmxData;
            softMaxParamNClasses->availableCmxBytes = availableCmxBytes;
            softMaxParamNClasses->ndims = lp->ndims;
            for (int32_t i = 0; i < MAX_ND_DIMS; i++) {
                softMaxParamNClasses->in_dims[i] = lp->in_dims[i];
                softMaxParamNClasses->in_strides[i] = lp->in_strides[i];
                softMaxParamNClasses->out_strides[i] = lp->out_strides[i];
            }
            softMaxParamNClasses->axis = lp->axis;
            softMaxParamNClasses->axisDim = lp->axisDim;
            softMaxParamNClasses->axisIStride = lp->axisIStride;
            softMaxParamNClasses->axisOStride = lp->axisOStride;
            softMaxParamNClasses->start = processed;
            softMaxParamNClasses->toProcess = to_process_on_shave;

            mvSoftMaxSingle(softMaxParamNClasses);
            processed += to_process_on_shave;
        }
    }
}

}
} // namespace shave_lib
} // namespace nn
