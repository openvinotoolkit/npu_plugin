// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernel_params.h>
#include <moviVectorUtils.h>
#include <algorithm>

void  CTCDecoder(
    const half *probabilities, const half *sequenceIndicators, half *output,
    int width, int height, int channels);

// default entry 0x1e
extern "C" void custom_entry(uint32_t* args, const KernelParams& kernelParams) {
    (void)kernelParams;

    CTCDecoder((const half*)args[0], (half*)args[1], (half*)args[2],
               (int)args[3], (int)args[4], (int)args[5]);
}

void  CTCDecoder(const half *probabilities,
                            const half *sequenceIndicators,
                            half *output,

                            int width,
                            int height,
                            int channels) {
    int T_ = channels;
    int B_ = height;
    int C_ = width;

    // Fill output with -1
    for (int i = 0; i < B_ * T_; i++)
    {
        output[i] = -1.h;
    }

    int output_index = 0;

    for(int b = 0; b < B_; ++b)
    {
        const half *seq_ind = sequenceIndicators + b*T_;
        int seq_len = std::find(seq_ind + 1, seq_ind + T_, 0.h) - seq_ind;
        int time = std::min(seq_len, T_);

        int prev_class_idx = -1;

        for (int t = 0; t < time; ++t)
        {
            int max_class_idx = 0;
            const half *probs = probabilities + b * C_ + t * C_ * B_;
            half max_prob = probs[0];

            for (int c = 1; c < C_; ++c)
            {
                if (probs[c] > max_prob)
                {
                    max_class_idx = c;
                    max_prob = probs[c];
                }
            }

            if (max_class_idx < C_-1 && max_class_idx != prev_class_idx)
            {
                output[b * T_ + output_index] = (half)max_class_idx;
                output_index++;
            }

            prev_class_idx = max_class_idx;
        }
    }
}
