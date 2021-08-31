// {% copyright %}

#include <mv_types.h>
#include <math.h>
#include <moviVectorTypes.h>
#include <moviVectorConvert.h>
#include "softMaxNClasses_core.h"
#include <mvSubspaces.h>
#include <dma_shave.h>

using namespace subspace;

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

static void wait_secondary_shaves(intershaves ** inter_data, int shaves_no)
{
    bool not_ready;
    do
    {
        not_ready = false;
        for(int i = 1; i < shaves_no; i++)
        {
            if(inter_data[i]->status != finished)
            {
                not_ready = true;
                break;
            }
        }
    }while(not_ready);
}

void mvSoftMaxInner_1x1xN(t_MvSoftMaxParamNClasses* p)
{
    u8 * input  = (u8 *)(p->input);
    u8 * output = (u8 *)(p->output);
    int primary_shave = p->grpLeaderShave;
    int shaves_no    = p->grpShavesNo;
    int this_shave   = p->this_shave;
    int dim2_shift = p->start;
    int elems_no = p->toProcess;

    u8 * slice0 = (u8*)(p->cmxslice);
    u8 * slice1 = (u8*)(p->auxcmxslice);
    intershaves *my_data = ((intershaves*)slice0) + (this_shave - primary_shave);
    intershaves * inter_data[CONFIG_SOC_SHAVE_COUNT];

    half* in  = (half*)(slice0 + CONFIG_SOC_SHAVE_COUNT * sizeof(intershaves)) + dim2_shift;
    half* out = (half*)(slice1 + CONFIG_SOC_SHAVE_COUNT * sizeof(intershaves*)) + dim2_shift;

    my_data->status = invalid;
    if(this_shave == primary_shave)
    {
        DmaAlShave dmaTask;
        dmaTask.start((u8*)(input), (u8*)in, p->axisDim * sizeof(half));
        dmaTask.wait();

        for(int i = 0; i < shaves_no; i++)
            inter_data[i] = ((intershaves *)slice0) + i;

        wait_secondary_shaves(inter_data, shaves_no);
        for(int i = 0; i < shaves_no; i++)
        {
            inter_data[i]->status = aggregated;
        }
    }
    else
    {
        int timeout = 0;
        my_data->status = finished;
        do {} while(my_data->status != aggregated && timeout++ < 1000000);
    }

    // stage 1: largest element finding
    my_data->status = working;

    half largest = largest_softmax_C(in, elems_no);

    my_data->largest = largest;
    my_data->status = finished;

    if(this_shave == primary_shave)
    {
        wait_secondary_shaves(inter_data, shaves_no);
        for(int i = 1; i < shaves_no; i++)
        {
            largest = __builtin_shave_cmu_max_f16_rr_half(largest, inter_data[i]->largest);
        }
        for(int i = 1; i < shaves_no; i++)
        {
            inter_data[i]->largest = largest;
            inter_data[i]->status = aggregated;
        }
    }
    else
    {
        do {} while(my_data->status != aggregated);
        largest = my_data->largest;
    }
    // end of stage 1

    // stage 2: exp, sum calculation
    my_data->status = working;
    float sumf = sum_softmax_C(in, elems_no, out, largest);

    my_data->sumf = sumf;
    my_data->status = finished;
    half reciprocal_sum;
    if(this_shave == primary_shave)
    {
        wait_secondary_shaves(inter_data, shaves_no);
        for(int i = 1; i < shaves_no; i++)
        {
            sumf += inter_data[i]->sumf;
        }
        if(!((half)sumf))
            reciprocal_sum = (half)INFINITY;
        else
            reciprocal_sum = (half)1.f/(half)sumf;
        for(int i = 1; i < shaves_no; i++)
        {
            inter_data[i]->reciprocal_sum = reciprocal_sum;
            inter_data[i]->status = aggregated;
        }
    }
    else
    {
        do {} while(my_data->status != aggregated);
        reciprocal_sum = my_data->reciprocal_sum;
    }
    // end of stage 2

    // stage 3: normalization
    my_data->status = working;
    {
        int i = 0;
        for(; i < elems_no; ++i)
        {
            out[i] *= reciprocal_sum;
        }
    }
    my_data->status = finished;

    if(this_shave == primary_shave)
    {
        wait_secondary_shaves(inter_data, shaves_no);
        DmaAlShave dmaTask;
        dmaTask.start((u8*)out, (u8*)(output), p->axisDim * sizeof(half));
        dmaTask.wait();
        for(int i = 0; i < shaves_no; i++)
        {
            inter_data[i]->status = invalid;
        }
    }
}

// softmax on inner/outer dimensions
void mvSoftMax(t_MvSoftMaxParamNClasses *p)
{
    const half* in  = p->input;
    half* out = p->output;

    s32* dims = p->in_dims;
    s32* istrides = p->in_strides;
    s32* ostrides = p->out_strides;
    s32 ndims = p->ndims;

    DmaAlShave dmaRTask;

    half* p_input0  = reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* p_output0 = reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);

    int sets_per_step = (WORK_BUFFER_SIZE) / (sizeof(half) * p->axisDim);
    int32_t setCoords[MAX_ND_DIMS];

    void (*calculate)(const int dimX, int n, half* input, half* output,
            int istride1, int istride2, int ostride1, int ostride2);
    calculate = (p->axis > 0) ? calculateSoftMaxOuter : calculateSoftMaxInner;

#ifdef MA2450
    int num_of_sets = p->toProcess ;
    half* p_input1  = reinterpret_cast<half*>(p->cmxslice + 1 * WORK_BUFFER_SIZE);
    half* p_output1 = reinterpret_cast<half*>(p->cmxslice + 3 * WORK_BUFFER_SIZE);
    if(num_of_sets / sets_per_step > 3 || num_of_sets / dims[0] > 2)
    {
        int i = p->start;
        int r_step0, r_step1, r_step2;
        int axisDim = p->axisDim;

        int* dmaWidth0;
        int* dmaWidth1;
        int* dmaWidth2;

        if(p->axis)
        {
            dmaWidth0 = &r_step0;
            dmaWidth1 = &r_step1;
            dmaWidth2 = &r_step2;
        }
        else
        {
            dmaWidth0 = &axisDim;
            dmaWidth1 = &axisDim;
            dmaWidth2 = &axisDim;
        }

        unsigned inOffset0, outOffset0, inOffset1, outOffset1;

        subspace::getCoord(i, dims, ndims, setCoords);
        {
            r_step0 = __builtin_shave_cmu_min_i32_rr_int(sets_per_step, dims[0] - setCoords[0]);

            subspace::getOffsetsU8(setCoords, istrides, ostrides, ndims, inOffset0, outOffset0);
            dmaRTask.start((u8*)in + inOffset0, (u8 *)p_input0,
                      sizeof(fp16) * p->axisDim * r_step0,
                      sizeof(fp16) * dmaWidth0[0],
                      sizeof(fp16) * dmaWidth0[0],
                      p->axisIStride,
                      sizeof(fp16) * dmaWidth0[0]);
            dmaRTask.wait();
            i += r_step0;
            subspace::incrementNCoord(setCoords, dims, ndims, r_step0);
        }

        {
            r_step1 = __builtin_shave_cmu_min_i32_rr_int(sets_per_step, dims[0] - setCoords[0]);

            subspace::getOffsetsU8(setCoords, istrides, ostrides, ndims, inOffset1, outOffset1);
            dmaRTask.start((u8*)in + inOffset1, (u8 *)p_input1,
                      sizeof(fp16) * p->axisDim * r_step1,
                      sizeof(fp16) * dmaWidth1[0],
                      sizeof(fp16) * dmaWidth1[0],
                      p->axisIStride,
                      sizeof(fp16) * dmaWidth1[0]);
            calculate(r_step0, p->axisDim, p_input0, p_output0,
                    1, r_step0, 1, r_step0);
            dmaRTask.wait();

            i += r_step1;
            subspace::incrementNCoord(setCoords, dims, ndims, r_step1);
        }

        while(i < p->start + p->toProcess)
        {
            unsigned inOffset2, outOffset2;
            r_step2 = __builtin_shave_cmu_min_i32_rr_int(sets_per_step, dims[0] - setCoords[0]);

            subspace::getOffsetsU8(setCoords, istrides, ostrides, ndims, inOffset2, outOffset2);
            dmaRTask.create((u8*)in + inOffset2, (u8 *)p_input0,
                      sizeof(fp16) * p->axisDim * r_step2,
                      sizeof(fp16) * dmaWidth2[0],
                      sizeof(fp16) * dmaWidth2[0],
                      /*sizeof(fp16) * */p->axisIStride,
                      sizeof(fp16) * dmaWidth2[0]);
            dmaWTask.create((u8 *)p_output0, (u8*)out + outOffset0,
                      sizeof(fp16) * p->axisDim * r_step0,
                      sizeof(fp16) * dmaWidth0[0],
                      sizeof(fp16) * dmaWidth0[0],
                      sizeof(fp16) * dmaWidth0[0],
                      p->axisOStride);
            dmaRTask.append(dmaWTask);
            dmaRTask.start();
            calculate(r_step1, p->axisDim, p_input1, p_output1,
                    1, r_step1, 1, r_step1);

            dmaRTask.wait();
            i += r_step2;
            subspace::incrementNCoord(setCoords, dims, ndims, r_step2);
            r_step0 = r_step1;
            r_step1 = r_step2;
            inOffset0 = inOffset1;
            inOffset1 = inOffset2;
            outOffset0 = outOffset1;
            outOffset1 = outOffset2;
            SWAP(p_input0, p_input1, phalf);
            SWAP(p_output0, p_output1, phalf);
        }
        {
            dmaWTask.start((u8 *)p_output0, (u8*)out + outOffset0,
                      sizeof(fp16) * p->axisDim * r_step0,
                      sizeof(fp16) * dmaWidth0[0],
                      sizeof(fp16) * dmaWidth0[0],
                      sizeof(fp16) * dmaWidth0[0],
                      p->axisOStride);
            calculate(r_step1, p->axisDim, p_input1, p_output1,
                    1, r_step1, 1, r_step1);

            dmaWTask.wait();
            r_step0 = r_step1;
            inOffset0 = inOffset1;
            outOffset0 = outOffset1;
            SWAP(p_input0, p_input1, phalf);
            SWAP(p_output0, p_output1, phalf);
        }
        {
            dmaWTask.start((u8 *)p_output0, (u8*)out + outOffset0,
                      sizeof(fp16) * p->axisDim * r_step0,
                      sizeof(fp16) * dmaWidth0[0],
                      sizeof(fp16) * dmaWidth0[0],
                      sizeof(fp16) * dmaWidth0[0],
                      p->axisOStride);
            dmaWTask.wait();
        }
    }
    else
#endif
    {
        sets_per_step = (2 * WORK_BUFFER_SIZE) / (sizeof(half) * p->axisDim);
        int i = p->start;
        subspace::getCoord(i, dims, ndims, setCoords);
        int r_step;
        int axisDim = p->axisDim;

        int* dmaWidth = (p->axis) ? &r_step : &axisDim;
        while(i < p->start + p->toProcess)
        {
            r_step = __builtin_shave_cmu_min_i32_rr_int(sets_per_step, dims[0] - setCoords[0]);
            unsigned inOffset, outOffset;
            subspace::getOffsetsU8(setCoords, istrides, ostrides, ndims, inOffset, outOffset);
            dmaRTask.start((u8*)in + inOffset, (u8 *)p_input0,
                      sizeof(fp16) * p->axisDim * r_step,
                      sizeof(fp16) * dmaWidth[0],
                      sizeof(fp16) * dmaWidth[0],
                      p->axisIStride,
                      sizeof(fp16) * dmaWidth[0]);
            dmaRTask.wait();
            calculate(r_step, p->axisDim, p_input0, p_output0,
                    1, r_step, 1, r_step);
            dmaRTask.start((u8 *)p_output0, (u8*)out + outOffset,
                      sizeof(fp16) * p->axisDim * r_step,
                      sizeof(fp16) * dmaWidth[0],
                      sizeof(fp16) * dmaWidth[0],
                      sizeof(fp16) * dmaWidth[0],
                      p->axisOStride);
            dmaRTask.wait();
            i += r_step;
            subspace::incrementNCoord(setCoords, dims, ndims, r_step);
        }
    }
}
