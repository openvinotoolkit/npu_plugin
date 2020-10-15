#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__global const half *find(__global const half *begin, __global const half *end, half value) {
    while (begin != end) {
        if (*begin == value)  {
            return begin;
        }
        ++begin;
    }
    return end;
}

#define USE_MANUAL_DMA

#ifdef USE_MANUAL_DMA

__kernel void __dma_preload_CTCDecoder(__global half *restrict probabilities,
                                       __global half *restrict sequence_indicators,
                                       __global half *restrict output_sequences,
                                       int width,
                                       int height,
                                       int channels,
                                       __local half *restrict local_src,
                                       __local half *restrict local_dst)
{
    WorkGroupDmaCreateStrideTransaction(
        probabilities, // src
        local_src, // dst
        width * sizeof(half), // src_width,
        width * sizeof(half), // dst_width,
        width * height * sizeof(half), // src_stride,
        width * height * sizeof(half), // dst_stride,
        width * height * channels * sizeof(half), // size
        0);
}

__kernel void __dma_postwrite_CTCDecoder(__global half *restrict probabilities,
                                         __global half *restrict sequence_indicators,
                                         __global half *restrict output_sequences,
                                         int width,
                                         int height,
                                         int channels,
                                         __local half *restrict local_src,
                                         __local half *restrict local_dst)
{
    WorkGroupDmaCreateStrideTransaction(
        local_dst, // src
        output_sequences, // dst
        channels * sizeof(half), // src_width,
        channels * sizeof(half), // dst_width,
        channels * sizeof(half), // src_stride,
        channels * sizeof(half), // dst_stride,
        channels * height * sizeof(half), // size
        0);
}

__kernel void CTCDecoder(__global half *restrict _0,
                         __global half *restrict sequence_indicators,
                         __global half *restrict _1,
                         int width,
                         int height,
                         int channels,
                         __local half *restrict local_src,
                         __local half *restrict local_dst)
{
    const int T_ = channels;
    const int B_ = height;
    const int C_ = width;

    #pragma unroll 8
    for (int i = 0; i < B_*T_; i++)
    {
        local_dst[i] = -1.h;
    }

    int output_index = 0;

    for (int b = 0; b < B_; ++b)
    {
        __global const half *seq_ind = sequence_indicators + b*T_;
        const int seq_len = find(seq_ind + 1, seq_ind + T_, 0.h) - seq_ind;
        const int time = min(seq_len, T_);

        int prev_class_idx = -1;

        for (int t = 0; t < time; ++t)
        {
            __local const half *probs = local_src + b*C_ + t*C_*B_;
            int max_class_idx = 0;
            half max_prob = probs[0];

            for (int c = 1; c < C_; ++c)
            {
                const half prob = probs[c];
                if (prob > max_prob)
                {
                    max_class_idx = c;
                    max_prob = prob;
                }
            }

            if (max_class_idx < C_-1 && max_class_idx != prev_class_idx)
            {
                local_dst[b*T_ + output_index] = (half)max_class_idx;
                output_index++;
            }

            prev_class_idx = max_class_idx;
        }
    }
}

#else

__kernel void CTCDecoder(__global half *probabilities,
                         __global half *sequence_indicators,
                         __global half *output_sequences,
                         int width,
                         int height,
                         int channels,
                         __local half *local_src,
                         __local half *local_dst)
{
    const int T_ = channels;
    const int B_ = height;
    const int C_ = width;

    for (int i = 0; i < B_*T_; i++)
    {
        output_sequences[i] = -1.h;
    }

    int output_index = 0;

    for (int b = 0; b < B_; ++b)
    {
        __global const half *seq_ind = sequence_indicators + b*T_;
        const int seq_len = find(seq_ind + 1, seq_ind + T_, 0.h) - seq_ind;
        const int time = min(seq_len, T_);

        int prev_class_idx = -1;

        for (int t = 0; t < time; ++t)
        {
            __global const half *probs = probabilities + b*C_ + t*C_*B_;
            int max_class_idx = 0;
            half max_prob = probs[0];

            for (int c = 1; c < C_; ++c)
            {
                const half prob = probs[c];
                if (prob > max_prob)
                {
                    max_class_idx = c;
                    max_prob = prob;
                }
            }

            if (max_class_idx < C_-1 && max_class_idx != prev_class_idx)
            {
                output_sequences[b*T_ + output_index] = (half)max_class_idx;
                output_index++;
            }

            prev_class_idx = max_class_idx;
        }
    }
}

#endif
