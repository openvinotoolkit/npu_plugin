// {% copyright %}

#include <algorithm>
#include <dma_shave.h>
#include <moviVectorUtils.h>
#include <param_permute.h>
#include <sw_shave_lib_common.h>
#include <nn_log.h>

namespace {

#define MIN(a, b) __builtin_shave_cmu_min_i32_rr_int((a), (b))
#define DIVIDE_UP(dividend, divisor) (((dividend) + (divisor) -1) / (divisor))

template<typename T>
static void Copy(T *in, T *out, unsigned int w, int in_innerstride,
                 unsigned int h, int out_innerstride) {
    for (unsigned int i = 0; i < h; ++i) {
        for (unsigned int j = 0; j < w; ++j) {
            out[(i * w + j) * out_innerstride] = in[(i * w + j) * in_innerstride];
        }
    }
}

static void Copy(void *in, void *out, unsigned int w, int in_innerstride,
                 unsigned int h, int out_innerstride, unsigned int bpp) {
    switch (bpp) {
        case 4:
            Copy<u32>((u32 *) in, (u32 *) out, w, in_innerstride, h, out_innerstride);
            break;
        case 2:
            Copy<u16>((u16 *) in, (u16 *) out, w, in_innerstride, h, out_innerstride);
            break;
        case 1:
            Copy<u8>((u8 *) in, (u8 *) out, w, in_innerstride, h, out_innerstride);
            break;
        default:
            nnLog(MVLOG_DEBUG, "Unsupported data type\n");
            break;
    }
}

template<typename T>
static void Transp(T *in, T *out, unsigned int w, int in_innerstride,
                   unsigned int h, int out_innerstride) {
    for (unsigned int i = 0; i < h; ++i) {
        for (unsigned int j = 0; j < w; ++j) {
            out[(j * h + i) * out_innerstride] = in[(i * w + j) * in_innerstride];
        }
    }
}

static void Transp(void *in, void *out, unsigned int w, int in_innerstride,
                   unsigned int h, int out_innerstride, unsigned int bpp) {
    switch (bpp) {
        case 4:
            Transp<u32>((u32 *) in, (u32 *) out, w, in_innerstride, h, out_innerstride);
            break;
        case 2:
            Transp<u16>((u16 *) in, (u16 *) out, w, in_innerstride, h, out_innerstride);
            break;
        case 1:
            Transp<u8>((u8 *) in, (u8 *) out, w, in_innerstride, h, out_innerstride);
            break;
        default:
            nnLog(MVLOG_DEBUG, "Unsupported data type\n");
            break;
    }
}

static inline void Tiling(int M, int w, int h, int innerstride, int &w_step, int &h_step) {
    int num_of_str = 1;
    int num_of_tiles_in_str = DIVIDE_UP(DIVIDE_UP(h, num_of_str) * (w * innerstride), M);
    while (DIVIDE_UP(w, num_of_tiles_in_str) < 8 && num_of_tiles_in_str > 3 && num_of_str < h) {
        num_of_str++;
        num_of_tiles_in_str = DIVIDE_UP(DIVIDE_UP(h, num_of_str) * (w * innerstride), M);
    }
    h_step = (h / num_of_str);
    w_step = (w / num_of_tiles_in_str);
}

}  // namespace

static void cmxTranspose_half(half* __restrict pinput, half* __restrict poutput, int width, int height, int channels)
{
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < height; h++)
        {
            int w = 0;
            for (; w < width - 15; w += 16)
            {
                half* pinp = pinput  + w + h * width  + c * width * height;
                half* pout = poutput + h + w * height + c * width * height;

                half8 v0 = *(half8*)(pinp + 0 * 8);
                half8 v1 = *(half8*)(pinp + 1 * 8);

                *(half*)(pout + 0 * height) = v0[0];
                *(half*)(pout + 1 * height) = v0[1];
                *(half*)(pout + 2 * height) = v0[2];
                *(half*)(pout + 3 * height) = v0[3];

                *(half*)(pout + 4 * height) = v0[4];
                *(half*)(pout + 5 * height) = v0[5];
                *(half*)(pout + 6 * height) = v0[6];
                *(half*)(pout + 7 * height) = v0[7];

                *(half*)(pout + 8  * height) = v1[0];
                *(half*)(pout + 9  * height) = v1[1];
                *(half*)(pout + 10 * height) = v1[2];
                *(half*)(pout + 11 * height) = v1[3];

                *(half*)(pout + 12 * height) = v1[4];
                *(half*)(pout + 13 * height) = v1[5];
                *(half*)(pout + 14 * height) = v1[6];
                *(half*)(pout + 15 * height) = v1[7];
            }
            for (; w < width; w++)
            {
                half* pinp = pinput  + w + h * width  + c * width * height;
                half* pout = poutput + h + w * height + c * width * height;

                *pout = *pinp;
            }
        }
    }
}

static void dma_create_3d(DmaAlShave& dma, const void* src, void* dst, u32 byteLength,
                          u32 srcWidth, u32 dstWidth, u32 srcStride, u32 dstStride,
                          u32 numPlanes, u32 srcPlaneStride, u32 dstPlaneStride)
{
    if (((byteLength % srcWidth) == 0) && (srcStride * byteLength == srcPlaneStride * srcWidth) &&
        ((byteLength % dstWidth) == 0) && (dstStride * byteLength == dstPlaneStride * dstWidth))
    {
        byteLength *= numPlanes;
        numPlanes = 1;
    }
    if (srcWidth == srcStride)
        srcWidth = srcStride = byteLength;
    if (dstWidth == dstStride)
        dstWidth = dstStride = byteLength;

    dma.start(src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride, numPlanes, srcPlaneStride, dstPlaneStride);
}

static void dma_create_2d(DmaAlShave& dma, const void* src, void* dst, u32 byteLength,
                          u32 srcWidth, u32 dstWidth, u32 srcStride, u32 dstStride)
{
    if (srcWidth == srcStride)
        srcWidth = srcStride = byteLength;
    if (dstWidth == dstStride)
        dstWidth = dstStride = byteLength;
    dma.start(src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride);
}

void mvPermuteTranspose(nn::shave_lib::t_PermTransposeParam *params)
{
    DmaAlShave task;

    const half* input = (half*)params->input;
    half* output = (half*)params->output;

    const auto bpp = sizeof(half);

    int width  = params->width;

    int batch0 = params->batch0;
    int batch1 = params->batch1;

    int height0 = params->height0;
    int height1 = params->height1;
    int height_size = (height1 - height0);
    if (height_size <= 0) return;

    int matrix_size = width * height_size;

    if ((int)(matrix_size * 2 * bpp) > (int)(params->cmxSize)) // if matrix is quite huge
    {
        int height_step = MIN((int)(params->cmxSize) / (2  * width * bpp), height_size);
        if (height_step == 0) // CMX memory is not enough
        {
            return;
        }
        int matrix_size = width * height_step;

        half* cmxInputBuffer  = (half*)params->cmxData;
        half* cmxOutputBuffer = cmxInputBuffer + matrix_size;

        int input_stride0  = params->stride_in_line;
        int input_stride1  = params->stride_in_batch;
        int output_stride0 = params->stride_out_line;
        int output_stride1 = params->stride_out_batch;

        for (int b = batch0; b < batch1; b++)
        {
            for (int h = height0; h < height1; h += height_step)
            {
                int height_step_real = MIN(height_step, height1 - h);

                dma_create_2d(task, (u8*)(input) + h * input_stride0 + b * input_stride1, (u8*)cmxInputBuffer,
                              height_step_real * width * bpp,
                              width * bpp,
                              width * bpp,
                              input_stride0,
                              width * bpp);
                task.wait();

                cmxTranspose_half(cmxInputBuffer, cmxOutputBuffer, width, height_step_real, 1);

                dma_create_2d(task, (u8*)cmxOutputBuffer,
                              (u8*)(output) + h * bpp + b * output_stride1,
                              height_step_real * width * bpp,

                              height_step_real * bpp,
                              height_step_real * bpp,
                              height_step_real * bpp,
                              output_stride0);
                task.wait();
            }
        }
    }
    else
    {
        int batch_step = (int)(params->cmxSize) / (matrix_size * bpp * 2);
        batch_step = MIN(batch_step, batch1 - batch0);

        half* cmxInputBuffer  = (half*)params->cmxData;
        half* cmxOutputBuffer = cmxInputBuffer + matrix_size * batch_step;

        int input_stride0  = params->stride_in_line;
        int input_stride1  = params->stride_in_batch;
        int output_stride0 = params->stride_out_line;
        int output_stride1 = params->stride_out_batch;

        for (int b = batch0; b < batch1; b += batch_step)
        {
            int batch_real_step = MIN(batch_step, batch1 - b);

            dma_create_3d(task, (u8*)(input) + b * input_stride1 + height0 * input_stride0, (u8*)cmxInputBuffer,
                          matrix_size * bpp,
                          width * bpp,
                          width * bpp,
                          input_stride0,
                          width * bpp,
                          batch_real_step, input_stride1, matrix_size * bpp);
            task.wait();

            cmxTranspose_half(cmxInputBuffer, cmxOutputBuffer, width, height_size, batch_real_step);

            dma_create_3d(task, (u8*)cmxOutputBuffer,
                          (u8*)(output) + b * output_stride1 + height0 * bpp,
                          matrix_size * bpp,
                          height_size * bpp,
                          height_size * bpp,
                          height_size * bpp,
                          output_stride0,
                          batch_real_step, matrix_size * bpp, output_stride1);
            task.wait();
        }
    }
}

void mvPermute(nn::shave_lib::t_PermParam *params)
{
    DmaAlShave task;

    const auto mvtensor_heap_data_size = params->cmxSize;
    nn::shave_lib::PermForDMA parsedPerm = *(params->parsedPerm);
    const auto input = params->input;
    auto output = params->output;

    const auto bpp = params->bpp;

    unsigned int slice = params->slice;
    unsigned int n_slices = params->n_slices;
    unsigned int in_slice_offset;
    unsigned int out_slice_offset;

    int W = parsedPerm.dims_in[0];
    int H = parsedPerm.dims_in[1];
    int sliceDivider = params->sliceDivider;

    int width = W;
    int height = DIVIDE_UP(H, sliceDivider);
    int width_step = 0;
    int height_step = height;

    if(W > H)
    {
        width = DIVIDE_UP(W, sliceDivider);
        height = H;
        width_step = width;
        height_step = 0;
    }

    unsigned int slice_coord[MAX_ND_DIMS - 2];
    auto in_innerstride  = parsedPerm.strides_in [0];
    auto out_innerstride = parsedPerm.strides_out[0];

    nnLog(MVLOG_DEBUG, "Entry: in width %d, in_height %d, in_st0 %d, in_st1 %d, out width %d, out height %d, out_st0 %d, out_st1 %d\n"
          , parsedPerm.dims_in[0], parsedPerm.dims_in[1], parsedPerm.strides_in[0], parsedPerm.strides_in[1]
          , parsedPerm.dims_out[0], parsedPerm.dims_out[1], parsedPerm.strides_out[0], parsedPerm.strides_out[1]);

    nnLog(MVLOG_DEBUG, "H %d, W %d, height %d, width %d, slice %d, n_slices %d\n"
          , H, W, height, width, slice, n_slices);

    int real_slice = slice / sliceDivider;
    for(int i = parsedPerm.ndims - 2 - 1, c_slice = real_slice; i >= 0; i--)
    {
        slice_coord[i] = c_slice / parsedPerm.slice_sizes[i];
        c_slice = c_slice - slice_coord[i] * parsedPerm.slice_sizes[i];
    }

    if(parsedPerm.transpose)
    {
        int M;
        int h_step, w_step;
        if(width > height)
        {
            M = DIVIDE_UP((in_innerstride * width * (mvtensor_heap_data_size / bpp)),
                          (in_innerstride * width + out_innerstride * width)) & (~1);
            Tiling(M, width, height, in_innerstride / bpp, w_step, h_step);
        }
        else
        {
            M = DIVIDE_UP((in_innerstride * height * (mvtensor_heap_data_size / bpp)),
                          (in_innerstride * height + out_innerstride * height)) & (~1);
            Tiling(M, height, width, in_innerstride / bpp, h_step, w_step);
        }

        u8 *in = params->cmxData;
        u8 *out = in + M * bpp;

        int sub_slice = slice % sliceDivider;
        int shift_width  = sub_slice * width_step;
        int shift_height = sub_slice * height_step;
        nnLog(MVLOG_DEBUG, "sub_slice %d, shift_width %d, shift_height %d, w_step %d, h_step %d, width_step %d, height_step %d\n"
              , sub_slice, shift_width, shift_height, w_step, h_step, width_step, height_step);
        {
            in_slice_offset = 0;
            out_slice_offset = 0;
            for(unsigned int d = 0; d < parsedPerm.ndims - 2; ++ d)
            {
                in_slice_offset += slice_coord[d] * parsedPerm.strides_in[d + 2];
                out_slice_offset += slice_coord[d] * parsedPerm.strides_out[d + 2];
            }
        }

        for(unsigned int c_slice = slice; c_slice < slice + n_slices; ++c_slice)
        {
            int height_to_work = MIN(height, H - shift_height);
            int width_to_work  = MIN(width,  W - shift_width);

            nnLog(MVLOG_DEBUG, "sub_slice %d, shift_width %d, shift_height %d, width_to_work %d, height_to_work %d\n"
                  , sub_slice, shift_width, shift_height, width_to_work, height_to_work);

            for(int h = 0; h < height_to_work; h += h_step)
            {
                int h_to_work = MIN(h_step, height_to_work - h);
                for(int w = 0; w < width_to_work; w += w_step)
                {
                    int w_to_work = MIN(w_step, width_to_work - w);
                    unsigned int in_offset  = in_slice_offset +
                                              (h + shift_height) * parsedPerm.strides_in[1] +
                                              (w + shift_width) * in_innerstride;
                    unsigned int out_offset = out_slice_offset +
                                              (w + shift_width) * parsedPerm.strides_out[1] +
                                              (h + shift_height)* out_innerstride;
                    task.start(
                            (u8*)(input + in_offset),
                            (u8*)in,
                            h_to_work * w_to_work * in_innerstride,
                            w_to_work * in_innerstride,
                            w_to_work * in_innerstride,
                            parsedPerm.strides_in[1],
                            w_to_work * in_innerstride);
                    task.wait();
                    Transp(in, out, w_to_work, in_innerstride / bpp, h_to_work, out_innerstride / bpp, bpp);
                    task.start(
                            (u8*)out,
                            (u8*)(output + out_offset),
                            h_to_work * w_to_work * out_innerstride,
                            h_to_work * out_innerstride,
                            h_to_work * out_innerstride,
                            h_to_work * out_innerstride,
                            parsedPerm.strides_out[1]);
                    task.wait();
                }
            }
            ++sub_slice;
            if(sub_slice == sliceDivider)
            {
                for(unsigned int d = 0, n_add = 1; d < parsedPerm.ndims - 2 && n_add == 1 ; ++ d)
                {
                    slice_coord[d] = (slice_coord[d] == parsedPerm.dims_out[d + 2] - 1) ? 0 : slice_coord[d] + 1;
                    n_add = (slice_coord[d] == 0) ? 1 : 0;
                }
                sub_slice = 0;
                shift_height = 0;
                shift_width = 0;

                {
                    in_slice_offset = 0;
                    out_slice_offset = 0;
                    for(unsigned int d = 0; d < parsedPerm.ndims - 2; ++ d)
                    {
                        in_slice_offset += slice_coord[d] * parsedPerm.strides_in[d + 2];
                        out_slice_offset += slice_coord[d] * parsedPerm.strides_out[d + 2];
                    }
                }
            }
            else
            {
                shift_height += height_step;
                shift_width += width_step;
            }
        }
    }
    else if(in_innerstride / bpp != 1 || out_innerstride / bpp != 1)
    {
        int M;
        int h_step, w_step;
        if(width > height)
        {
            M = DIVIDE_UP((in_innerstride * width * (mvtensor_heap_data_size / bpp)),
                          (in_innerstride * width + out_innerstride * width)) & (~1);
            Tiling(M, width, height, in_innerstride / bpp, w_step, h_step);
        }
        else
        {
            M = DIVIDE_UP((in_innerstride * height * (mvtensor_heap_data_size / bpp)),
                          (in_innerstride * height + out_innerstride * height)) & (~1);
            Tiling(M, height, width, in_innerstride / bpp, h_step, w_step);
        }
        u8 *in = params->cmxData;
        u8 *out = in + M * bpp;

        int sub_slice = slice % sliceDivider;
        int shift_width  = sub_slice * width_step;
        int shift_height = sub_slice * height_step;
        nnLog(MVLOG_DEBUG, "sub_slice %d, shift_width %d, shift_height %d, w_step %d, h_step %d, width_step %d, height_step %d\n"
              , sub_slice, shift_width, shift_height, w_step, h_step, width_step, height_step);
        {
            in_slice_offset = 0;
            out_slice_offset = 0;
            for(unsigned int d = 0; d < parsedPerm.ndims - 2; ++ d)
            {
                in_slice_offset += slice_coord[d] * parsedPerm.strides_in[d + 2];
                out_slice_offset += slice_coord[d] * parsedPerm.strides_out[d + 2];
            }
        }

        for(unsigned int c_slice = slice; c_slice < slice + n_slices; ++c_slice)
        {
            int height_to_work = MIN(height, H - shift_height);
            int width_to_work  = MIN(width,  W - shift_width);

            nnLog(MVLOG_DEBUG, "sub_slice %d, shift_width %d, shift_height %d, width_to_work %d, height_to_work %d\n"
                  , sub_slice, shift_width, shift_height, width_to_work, height_to_work);

            nnLog(MVLOG_DEBUG, "not transpose M %d, width %d, height %d, in_innerstride %d, out_innerstride %d, w_step %u, h_step %u\n"
                    , M, width, height, in_innerstride, out_innerstride, w_step, h_step);

            for(int h = 0; h < height_to_work; h += h_step)
            {
                int h_to_work = MIN(h_step, height_to_work - h);
                for(int w = 0; w < width_to_work; w += w_step)
                {
                    int w_to_work = MIN(w_step, width_to_work - w);
                    unsigned int in_offset  = in_slice_offset +
                                              (h + shift_height) * parsedPerm.strides_in[1] +
                                              (w + shift_width) * in_innerstride;
                    unsigned int out_offset = out_slice_offset +
                                              (h + shift_height) * parsedPerm.strides_out[1] +
                                              (w + shift_width) * out_innerstride;
                    task.start(
                            (u8*)(input + in_offset),
                            (u8*)in,
                            h_to_work * w_to_work * in_innerstride,
                            w_to_work * in_innerstride,
                            w_to_work * in_innerstride,
                            parsedPerm.strides_in[1],
                            w_to_work * in_innerstride);
                    task.wait();
                    Copy(in, out, w_to_work, in_innerstride / bpp, h_to_work, out_innerstride / bpp, bpp);
                    task.start(
                            (u8*)out,
                            (u8*)(output + out_offset),
                            h_to_work * w_to_work * out_innerstride,
                            w_to_work * out_innerstride,
                            w_to_work * out_innerstride,
                            w_to_work * out_innerstride,
                            parsedPerm.strides_out[1]);
                    task.wait();
                }
            }
            ++sub_slice;
            if(sub_slice == sliceDivider)
            {
                for(unsigned int d = 0, n_add = 1; d < parsedPerm.ndims - 2 && n_add == 1 ; ++ d)
                {
                    slice_coord[d] = (slice_coord[d] == parsedPerm.dims_out[d + 2] - 1) ? 0 : slice_coord[d] + 1;
                    n_add = (slice_coord[d] == 0) ? 1 : 0;
                }
                sub_slice = 0;
                shift_height = 0;
                shift_width = 0;

                {
                    in_slice_offset = 0;
                    out_slice_offset = 0;
                    for(unsigned int d = 0; d < parsedPerm.ndims - 2; ++ d)
                    {
                        in_slice_offset += slice_coord[d] * parsedPerm.strides_in[d + 2];
                        out_slice_offset += slice_coord[d] * parsedPerm.strides_out[d + 2];
                    }
                }
            }
            else
            {
                shift_height += height_step;
                shift_width += width_step;
            }
        }
    }
    else
    {
        int sub_slice = slice % sliceDivider;
        int shift_width  = sub_slice * width_step;
        int shift_height = sub_slice * height_step;
        nnLog(MVLOG_DEBUG, "sub_slice %d, shift_width %d, shift_height %d, width_step %d, height_step %d\n"
              , sub_slice, shift_width, shift_height, width_step, height_step);
        {
            in_slice_offset = 0;
            out_slice_offset = 0;
            for(unsigned int d = 0; d < parsedPerm.ndims - 2; ++ d)
            {
                in_slice_offset += slice_coord[d] * parsedPerm.strides_in[d + 2];
                out_slice_offset += slice_coord[d] * parsedPerm.strides_out[d + 2];
            }
        }

        for(unsigned int c_slice = slice; c_slice < slice + n_slices; ++c_slice)
        {
            int height_to_work = MIN(height, H - shift_height);
            int width_to_work  = MIN(width,  W - shift_width);

            nnLog(MVLOG_DEBUG, "Not transpose\n");
            nnLog(MVLOG_DEBUG, "sub_slice %d, shift_width %d, shift_height %d, width_to_work %d, height_to_work %d\n"
                  , sub_slice, shift_width, shift_height, width_to_work, height_to_work);
            unsigned int in_offset  = in_slice_offset +
                                      shift_height * parsedPerm.strides_in[1] +
                                      shift_width * bpp;
            unsigned int out_offset = out_slice_offset +
                                      shift_height * parsedPerm.strides_out[1] +
                                      shift_width * bpp;
            task.start(
                    (u8*)(input + in_offset),
                    (u8*)(output + out_offset),
                    width_to_work  * height_to_work * bpp,
                    width_to_work * bpp,
                    width_to_work * bpp,
                    parsedPerm.strides_in[1],
                    parsedPerm.strides_out[1]);
            task.wait();

            ++sub_slice;
            if(sub_slice == sliceDivider)
            {
                for(unsigned int d = 0, n_add = 1; d < parsedPerm.ndims - 2 && n_add == 1 ; ++ d)
                {
                    slice_coord[d] = (slice_coord[d] == parsedPerm.dims_out[d + 2] - 1) ? 0 : slice_coord[d] + 1;
                    n_add = (slice_coord[d] == 0) ? 1 : 0;
                }
                sub_slice = 0;
                shift_height = 0;
                shift_width = 0;

                {
                    in_slice_offset = 0;
                    out_slice_offset = 0;
                    for(unsigned int d = 0; d < parsedPerm.ndims - 2; ++ d)
                    {
                        in_slice_offset += slice_coord[d] * parsedPerm.strides_in[d + 2];
                        out_slice_offset += slice_coord[d] * parsedPerm.strides_out[d + 2];
                    }
                }
            }
            else
            {
                shift_height += height_step;
                shift_width += width_step;
            }
        }
    }
}

extern "C" {
void nnPermute(nn::shave_lib::mvPermuteParams *params) {
    if (params->is_shave_enabled){
        if (params->run_mv_transpose) {
            mvPermuteTranspose(&params->mvPermuteUnion.mvPermTransposeParam);
        } else {
            mvPermute(&params->mvPermuteUnion.mvPermParam);
        }
    }
}
}
