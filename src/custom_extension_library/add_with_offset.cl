// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

__kernel void add_w_offset(
    __global half *inp1,
    __global half *inp2,
    __global half *outp,
    int width,
    int height,
    int channels,
    float offset)
{
    const int T_ = channels;
    const int B_ = height;
    const int C_ = width;

    for (int i = 0; i < B_ * T_ * C_; i++) {
        outp[i] = inp1[i] + inp2[i] + offset;
    }
}
