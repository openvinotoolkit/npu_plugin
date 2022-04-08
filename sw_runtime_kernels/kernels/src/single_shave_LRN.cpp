//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorConvert.h>
#include <mv_types.h>

#include <param_lrn.h>

#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define intrinsic_vau_vec(intrinsic, vin, vout) (vout) = intrinsic((vin))
#define intrinsic_vec(intrinsic, vin, vout) \
    (vout)[0] = intrinsic((vin)[0]);        \
    (vout)[1] = intrinsic((vin)[1]);        \
    (vout)[2] = intrinsic((vin)[2]);        \
    (vout)[3] = intrinsic((vin)[3]);        \
    (vout)[4] = intrinsic((vin)[4]);        \
    (vout)[5] = intrinsic((vin)[5]);        \
    (vout)[6] = intrinsic((vin)[6]);        \
    (vout)[7] = intrinsic((vin)[7]);

#define sau_log_vec(vin, vout) intrinsic_vec(__builtin_shave_sau_log_f16_r, vin, vout)
#define sau_exp_vec(vin, vout) intrinsic_vec(__builtin_shave_sau_exp_f16_r, vin, vout)

#define vau_log_vec(vin, vout) (intrinsic_vau_vec(__builtin_shave_vau_log_v8f16_r, vin, vout))
#define vau_exp_vec(vin, vout) (intrinsic_vau_vec(__builtin_shave_vau_exp_v8f16_r, vin, vout))

using namespace sw_params;

namespace {
void check_axis(int32_t* p_act_axis, int32_t* axis_flag, int32_t numInputDims, int32_t numAxisElements) {
    for (int32_t i = 0; i < numInputDims; ++i) {
        axis_flag[i] = 0;
        for (int32_t j = 0; j < numAxisElements; ++j) {
            if (i == (p_act_axis[j]) || i == (p_act_axis[j] + numInputDims)) {
                axis_flag[i] = 1;
                break;
            }
        }
    }
}
void calculate_3d_first_axis(half* square_input_s, half* tmp_out_s, int32_t size, int32_t* inputDims,
                             int32_t numInputElements) {
    half pre_out[inputDims[1]][inputDims[0]];
    int32_t pre_from[inputDims[1]];
    int32_t pre_to[inputDims[1]];
    for (int32_t fir = 0; fir < inputDims[2]; ++fir) {
        for (int32_t sec = 0; sec < inputDims[1]; ++sec) {
            int32_t from = MAX(fir - (size - 1) / 2, 0);
            int32_t to = MIN(fir + (size - 1) / 2 + 1, inputDims[2]);

            half* out = tmp_out_s + fir * inputDims[1] * inputDims[0] + sec * inputDims[0];

            if (fir == 0) {
                for (int32_t pos = from; pos < to; ++pos) {
                    half* in = square_input_s + pos * inputDims[1] * inputDims[0] + sec * inputDims[0];
                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        ((half8*)out)[last] += ((half8*)in)[last];
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        out[last] += in[last];
                    }
                }
                pre_from[sec] = from;
                pre_to[sec] = to;
                for (int32_t i = 0; i < inputDims[0]; ++i) {
                    pre_out[sec][i] = out[i];
                }

            } else {
                if (pre_from[sec] != from) {
                    int32_t pos = pre_from[sec];
                    half* in = square_input_s + pos * inputDims[1] * inputDims[0] + sec * inputDims[0];

                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        ((half8*)out)[last] = ((half8*)pre_out[sec])[last] - ((half8*)in)[last];
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        out[last] = pre_out[sec][last] - in[last];
                    }
                }
                if (pre_to[sec] != to) {
                    int32_t pos = pre_to[sec];
                    half* in = square_input_s + pos * inputDims[1] * inputDims[0] + sec * inputDims[0];

                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        if (pre_from[sec] != from) {
                            ((half8*)out)[last] = ((half8*)out)[last] + ((half8*)in)[last];
                        } else {
                            ((half8*)out)[last] = ((half8*)pre_out[sec])[last] + ((half8*)in)[last];
                        }
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        if (pre_from[sec] != from) {
                            out[last] = out[last] + in[last];
                        } else {
                            out[last] = pre_out[sec][last] + in[last];
                        }
                    }
                }
                if (pre_from[sec] == from && pre_to[sec] == to) {
                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        ((half8*)out)[last] = ((half8*)pre_out[sec])[last];
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        out[last] = pre_out[sec][last];
                    }
                }
                pre_from[sec] = from;
                pre_to[sec] = to;
                for (int32_t i = 0; i < inputDims[0]; ++i) {
                    pre_out[sec][i] = out[i];
                }
            }
        }
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        square_input_s[i] = tmp_out_s[i];
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
}
void calculate_3d_second_axis(half* square_input_s, half* tmp_out_s, int32_t size, int32_t* inputDims,
                              int32_t numInputElements) {
    half pre_out[inputDims[0]];
    int32_t pre_from;
    int32_t pre_to;
    for (int32_t fir = 0; fir < inputDims[2]; ++fir) {
        for (int32_t sec = 0; sec < inputDims[1]; ++sec) {
            int32_t from = MAX(sec - (size - 1) / 2, 0);
            int32_t to = MIN(sec + (size - 1) / 2 + 1, inputDims[1]);
            half* out = tmp_out_s + fir * inputDims[1] * inputDims[0] + sec * inputDims[0];

            if (sec == 0) {
                for (int32_t pos = from; pos < to; ++pos) {
                    half* in = square_input_s + fir * inputDims[1] * inputDims[0] + pos * inputDims[0];
                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        ((half8*)out)[last] += ((half8*)in)[last];
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        out[last] += in[last];
                    }
                }

                pre_from = from;
                pre_to = to;
                for (int32_t i = 0; i < inputDims[0]; ++i) {
                    pre_out[i] = out[i];
                }

            } else {
                if (pre_from != from) {
                    int32_t pos = pre_from;
                    half* in = square_input_s + fir * inputDims[1] * inputDims[0] + pos * inputDims[0];
                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        ((half8*)out)[last] = ((half8*)pre_out)[last] - ((half8*)in)[last];
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        out[last] = pre_out[last] - in[last];
                    }
                }
                if (pre_to != to) {
                    int32_t pos = pre_to;
                    half* in = square_input_s + fir * inputDims[1] * inputDims[0] + pos * inputDims[0];
                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        if (pre_from != from) {
                            ((half8*)out)[last] = ((half8*)out)[last] + ((half8*)in)[last];
                        } else {
                            ((half8*)out)[last] = ((half8*)pre_out)[last] + ((half8*)in)[last];
                        }
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        if (pre_from != from) {
                            out[last] = out[last] + in[last];
                        } else {
                            out[last] = pre_out[last] + in[last];
                        }
                    }
                }
                if (pre_from == from && pre_to == to) {
                    int32_t last = 0;
                    for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                        ((half8*)out)[last] = ((half8*)pre_out)[last];
                    }
                    for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                        out[last] = pre_out[last];
                    }
                }

                pre_from = from;
                pre_to = to;
                for (int32_t i = 0; i < inputDims[0]; ++i) {
                    pre_out[i] = out[i];
                }
            }
        }
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        square_input_s[i] = tmp_out_s[i];
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
}
void calculate_3d_last_axis(half* square_input_s, half* tmp_out_s, int32_t size, int32_t* inputDims,
                            int32_t numInputElements) {
    half pre_out;
    int32_t pre_from;
    int32_t pre_to;
    for (int32_t fir = 0; fir < inputDims[2]; ++fir) {
        for (int32_t sec = 0; sec < inputDims[1]; ++sec) {
            for (int32_t last = 0; last < inputDims[0]; ++last) {
                int32_t from = MAX(last - (size - 1) / 2, 0);
                int32_t to = MIN(last + (size - 1) / 2 + 1, inputDims[0]);
                half* out = tmp_out_s + fir * inputDims[1] * inputDims[0] + sec * inputDims[0];
                half* in = square_input_s + fir * inputDims[1] * inputDims[0] + sec * inputDims[0];

                if (last == 0) {
                    for (int32_t pos = from; pos < to; ++pos) {
                        out[last] += in[pos];
                    }
                    pre_from = from;
                    pre_to = to;
                    pre_out = out[last];
                } else {
                    if (pre_from != from) {
                        out[last] = pre_out - in[pre_from];
                    }
                    if (pre_to != to) {
                        if (pre_from != from) {
                            out[last] = out[last] + in[pre_to];
                        } else {
                            out[last] = pre_out + in[pre_to];
                        }
                    }
                    if (pre_from == from && pre_to == to) {
                        out[last] = pre_out;
                    }
                    pre_from = from;
                    pre_to = to;
                    pre_out = out[last];
                }
            }
        }
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        square_input_s[i] = tmp_out_s[i];
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
}

void calculate_4d_first_axis(half* square_input_s, half* tmp_out_s, int32_t size, int32_t* inputDims,
                             int32_t numInputElements) {
    half pre_out[inputDims[2]][inputDims[1]][inputDims[0]];
    int32_t pre_from[inputDims[2]][inputDims[1]];
    int32_t pre_to[inputDims[2]][inputDims[1]];
    for (int32_t fir = 0; fir < inputDims[3]; ++fir) {
        for (int32_t sec = 0; sec < inputDims[2]; ++sec) {
            for (int32_t thi = 0; thi < inputDims[1]; ++thi) {
                int32_t from = MAX(fir - (size - 1) / 2, 0);
                int32_t to = MIN(fir + (size - 1) / 2 + 1, inputDims[3]);

                half* out = tmp_out_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                            sec * inputDims[1] * inputDims[0] + thi * inputDims[0];

                if (fir == 0) {
                    for (int32_t pos = from; pos < to; ++pos) {
                        half* in = square_input_s + pos * inputDims[2] * inputDims[1] * inputDims[0] +
                                   sec * inputDims[1] * inputDims[0] + thi * inputDims[0];
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] += ((half8*)in)[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] += in[last];
                        }
                    }
                    pre_from[sec][thi] = from;
                    pre_to[sec][thi] = to;
                    for (int32_t i = 0; i < inputDims[0]; ++i) {
                        pre_out[sec][thi][i] = out[i];
                    }
                } else {
                    if (pre_from[sec][thi] != from) {
                        int32_t pos = pre_from[sec][thi];
                        half* in = square_input_s + pos * inputDims[2] * inputDims[1] * inputDims[0] +
                                   sec * inputDims[1] * inputDims[0] + thi * inputDims[0];

                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] = ((half8*)pre_out[sec][thi])[last] - ((half8*)in)[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] = pre_out[sec][thi][last] - in[last];
                        }
                    }
                    if (pre_to[sec][thi] != to) {
                        int32_t pos = pre_to[sec][thi];
                        half* in = square_input_s + pos * inputDims[2] * inputDims[1] * inputDims[0] +
                                   sec * inputDims[1] * inputDims[0] + thi * inputDims[0];

                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            if (pre_from[sec][thi] != from) {
                                ((half8*)out)[last] = ((half8*)out)[last] + ((half8*)in)[last];
                            } else {
                                ((half8*)out)[last] = ((half8*)pre_out[sec][thi])[last] + ((half8*)in)[last];
                            }
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            if (pre_from[sec][thi] != from) {
                                out[last] = out[last] + in[last];
                            } else {
                                out[last] = pre_out[sec][thi][last] + in[last];
                            }
                        }
                    }
                    if (pre_from[sec][thi] == from && pre_to[sec][thi] == to) {
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] = ((half8*)pre_out[sec][thi])[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] = pre_out[sec][thi][last];
                        }
                    }
                    pre_from[sec][thi] = from;
                    pre_to[sec][thi] = to;
                    for (int32_t i = 0; i < inputDims[0]; ++i) {
                        pre_out[sec][thi][i] = out[i];
                    }
                }
            }
        }
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        square_input_s[i] = tmp_out_s[i];
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
}
void calculate_4d_second_axis(half* square_input_s, half* tmp_out_s, int32_t size, int32_t* inputDims,
                              int32_t numInputElements) {
    half pre_out[inputDims[1]][inputDims[0]];
    int32_t pre_from[inputDims[1]];
    int32_t pre_to[inputDims[1]];
    for (int32_t fir = 0; fir < inputDims[3]; ++fir) {
        for (int32_t sec = 0; sec < inputDims[2]; ++sec) {
            for (int32_t thi = 0; thi < inputDims[1]; ++thi) {
                int32_t from = MAX(sec - (size - 1) / 2, 0);
                int32_t to = MIN(sec + (size - 1) / 2 + 1, inputDims[2]);
                half* out = tmp_out_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                            sec * inputDims[1] * inputDims[0] + thi * inputDims[0];

                if (sec == 0) {
                    for (int32_t pos = from; pos < to; ++pos) {
                        half* in = square_input_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                                   pos * inputDims[1] * inputDims[0] + thi * inputDims[0];
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] += ((half8*)in)[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] += in[last];
                        }
                    }

                    pre_from[thi] = from;
                    pre_to[thi] = to;
                    for (int32_t i = 0; i < inputDims[0]; ++i) {
                        pre_out[thi][i] = out[i];
                    }

                } else {
                    if (pre_from[thi] != from) {
                        int32_t pos = pre_from[thi];
                        half* in = square_input_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                                   pos * inputDims[1] * inputDims[0] + thi * inputDims[0];
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] = ((half8*)pre_out[thi])[last] - ((half8*)in)[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] = pre_out[thi][last] - in[last];
                        }
                    }
                    if (pre_to[thi] != to) {
                        int32_t pos = pre_to[thi];
                        half* in = square_input_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                                   pos * inputDims[1] * inputDims[0] + thi * inputDims[0];
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            if (pre_from[thi] != from) {
                                ((half8*)out)[last] = ((half8*)out)[last] + ((half8*)in)[last];
                            } else {
                                ((half8*)out)[last] = ((half8*)pre_out[thi])[last] + ((half8*)in)[last];
                            }
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            if (pre_from[thi] != from) {
                                out[last] = out[last] + in[last];
                            } else {
                                out[last] = pre_out[thi][last] + in[last];
                            }
                        }
                    }
                    if (pre_from[thi] == from && pre_to[thi] == to) {
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] = ((half8*)pre_out[thi])[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] = pre_out[thi][last];
                        }
                    }

                    pre_from[thi] = from;
                    pre_to[thi] = to;
                    for (int32_t i = 0; i < inputDims[0]; ++i) {
                        pre_out[thi][i] = out[i];
                    }
                }
            }
        }
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        square_input_s[i] = tmp_out_s[i];
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
}
void calculate_4d_third_axis(half* square_input_s, half* tmp_out_s, int32_t size, int32_t* inputDims,
                             int32_t numInputElements) {
    half pre_out[inputDims[0]];
    int32_t pre_from;
    int32_t pre_to;
    for (int32_t fir = 0; fir < inputDims[3]; ++fir) {
        for (int32_t sec = 0; sec < inputDims[2]; ++sec) {
            for (int32_t thi = 0; thi < inputDims[1]; ++thi) {
                int32_t from = MAX(thi - (size - 1) / 2, 0);
                int32_t to = MIN(thi + (size - 1) / 2 + 1, inputDims[1]);
                half* out = tmp_out_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                            sec * inputDims[1] * inputDims[0] + thi * inputDims[0];

                if (thi == 0) {
                    for (int32_t pos = from; pos < to; ++pos) {
                        half* in = square_input_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                                   sec * inputDims[1] * inputDims[0] + pos * inputDims[0];
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] += ((half8*)in)[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] += in[last];
                        }
                    }

                    pre_from = from;
                    pre_to = to;
                    for (int32_t i = 0; i < inputDims[0]; ++i) {
                        pre_out[i] = out[i];
                    }

                } else {
                    if (pre_from != from) {
                        int32_t pos = pre_from;
                        half* in = square_input_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                                   sec * inputDims[1] * inputDims[0] + pos * inputDims[0];
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] = ((half8*)pre_out)[last] - ((half8*)in)[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] = pre_out[last] - in[last];
                        }
                    }
                    if (pre_to != to) {
                        int32_t pos = pre_to;
                        half* in = square_input_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                                   sec * inputDims[1] * inputDims[0] + pos * inputDims[0];
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            if (pre_from != from) {
                                ((half8*)out)[last] = ((half8*)out)[last] + ((half8*)in)[last];
                            } else {
                                ((half8*)out)[last] = ((half8*)pre_out)[last] + ((half8*)in)[last];
                            }
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            if (pre_from != from) {
                                out[last] = out[last] + in[last];
                            } else {
                                out[last] = pre_out[last] + in[last];
                            }
                        }
                    }
                    if (pre_from == from && pre_to == to) {
                        int32_t last = 0;
                        for (; last < inputDims[0] / VECTOR_SIZE; ++last) {
                            ((half8*)out)[last] = ((half8*)pre_out)[last];
                        }
                        for (last = last * VECTOR_SIZE; last < inputDims[0]; ++last) {
                            out[last] = pre_out[last];
                        }
                    }

                    pre_from = from;
                    pre_to = to;
                    for (int32_t i = 0; i < inputDims[0]; ++i) {
                        pre_out[i] = out[i];
                    }
                }
            }
        }
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        square_input_s[i] = tmp_out_s[i];
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
}
void calculate_4d_last_axis(half* square_input_s, half* tmp_out_s, int32_t size, int32_t* inputDims,
                            int32_t numInputElements) {
    half pre_out;
    int32_t pre_from;
    int32_t pre_to;
    for (int32_t fir = 0; fir < inputDims[3]; ++fir) {
        for (int32_t sec = 0; sec < inputDims[2]; ++sec) {
            for (int32_t thi = 0; thi < inputDims[1]; ++thi) {
                for (int32_t last = 0; last < inputDims[0]; ++last) {
                    int32_t from = MAX(last - (size - 1) / 2, 0);
                    int32_t to = MIN(last + (size - 1) / 2 + 1, inputDims[0]);
                    half* out = tmp_out_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                                sec * inputDims[1] * inputDims[0] + thi * inputDims[0];
                    half* in = square_input_s + fir * inputDims[2] * inputDims[1] * inputDims[0] +
                               sec * inputDims[1] * inputDims[0] + thi * inputDims[0];
                    ;

                    if (last == 0) {
                        for (int32_t pos = from; pos < to; ++pos) {
                            out[last] += in[pos];
                        }
                        pre_from = from;
                        pre_to = to;
                        pre_out = out[last];
                    } else {
                        if (pre_from != from) {
                            out[last] = pre_out - in[pre_from];
                        }
                        if (pre_to != to) {
                            if (pre_from != from) {
                                out[last] = out[last] + in[pre_to];
                            } else {
                                out[last] = pre_out + in[pre_to];
                            }
                        }
                        if (pre_from == from && pre_to == to) {
                            out[last] = pre_out;
                        }
                        pre_from = from;
                        pre_to = to;
                        pre_out = out[last];
                    }
                }
            }
        }
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        square_input_s[i] = tmp_out_s[i];
    }
    for (int32_t i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
}

void calculate_lrn(half* square_input_s, int32_t* p_act_axis, half* tmp_out_s, int32_t size, int32_t* inputDims,
                   int32_t numInputElements, int32_t numAxisElements, int32_t numInputDims) {
    int32_t axis_flag[numInputDims];
    check_axis(p_act_axis, axis_flag, numInputDims, numAxisElements);

    if (numInputDims == 3) {
        for (int32_t axis = 0; axis < numInputDims; ++axis) {
            if (axis_flag[axis] == 1 && axis == 0) {
                calculate_3d_first_axis(square_input_s, tmp_out_s, size, inputDims, numInputElements);
            } else if (axis_flag[axis] == 1 && axis == 1) {
                calculate_3d_second_axis(square_input_s, tmp_out_s, size, inputDims, numInputElements);
            } else if (axis_flag[axis] == 1 && axis == 2) {
                calculate_3d_last_axis(square_input_s, tmp_out_s, size, inputDims, numInputElements);
            }
        }
    } else if (numInputDims == 4) {
        for (int32_t axis = 0; axis < numInputDims; ++axis) {
            if (axis_flag[axis] == 1 && axis == 0) {
                calculate_4d_first_axis(square_input_s, tmp_out_s, size, inputDims, numInputElements);
            } else if (axis_flag[axis] == 1 && axis == 1) {
                calculate_4d_second_axis(square_input_s, tmp_out_s, size, inputDims, numInputElements);
            } else if (axis_flag[axis] == 1 && axis == 2) {
                calculate_4d_third_axis(square_input_s, tmp_out_s, size, inputDims, numInputElements);
            } else if (axis_flag[axis] == 1 && axis == 3) {
                calculate_4d_last_axis(square_input_s, tmp_out_s, size, inputDims, numInputElements);
            }
        }
    }
}
}  // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void single_shave_LRN(const struct LRNParams* lParams) {
    half8* p_act_input_v = (half8*)(lParams->input.dataAddr);
    half* p_act_input_s = (half*)(lParams->input.dataAddr);
    int32_t* p_act_axis = (int32_t*)(lParams->axis.dataAddr);
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    float alpha = (float)lParams->alpha;
    float beta = (float)lParams->beta;
    float bias = (float)lParams->bias;
    int32_t size = (int32_t)lParams->size;

    int32_t* inputDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t* axisDims = (int32_t*)(lParams->axis.dimsAddr);

    int32_t numInputElements = 1;
    int32_t numAxisElements = axisDims[0];
    int32_t numInputDims = (int32_t)(lParams->input.numDims);

    for (int i = 0; i != numInputDims; i++) {
        numInputElements *= inputDims[i];
    }

    const int numVectors = numInputElements / VECTOR_SIZE;

    half square_input_s[numInputElements];
    half8* square_input_v = (half8*)(square_input_s);

#pragma clang loop unroll_count(8)
    for (int i = 0; i < numVectors; i++) {
        square_input_v[i] = p_act_input_v[i] * p_act_input_v[i];
    }
    for (int i = numVectors * VECTOR_SIZE; i < numInputElements; i++) {
        square_input_s[i] = p_act_input_s[i] * p_act_input_s[i];
    }

    half tmp_out_s[numInputElements];
    for (int i = 0; i < numInputElements; ++i) {
        tmp_out_s[i] = 0;
    }
    half8* tmp_out_v = (half8*)(tmp_out_s);

    if (size != 1 && size != 2) {
        calculate_lrn(square_input_s, p_act_axis, tmp_out_s, size, inputDims, numInputElements, numAxisElements,
                      numInputDims);
    }

    half scale = (half)numAxisElements * __builtin_shave_sau_log_f16_r((half)size);
    scale = 1.0f / __builtin_shave_sau_exp_f16_r(scale);

#pragma clang loop unroll_count(8)
    for (int i = 0; i < numVectors; i++) {
        half8 vout = square_input_v[i];
        vout = (half)(bias) + (alpha * vout) * scale;
        sau_log_vec(vout, vout);
        sau_exp_vec(beta * vout, vout);
        p_act_out_v[i] = p_act_input_v[i] / vout;
    }
    for (int i = numVectors * VECTOR_SIZE; i < numInputElements; i++) {
        half out = square_input_s[i];
        out = (half)(bias) + (alpha * out) * scale;
        out = __builtin_shave_sau_log_f16_r(out);
        out = __builtin_shave_sau_exp_f16_r(beta * out);
        p_act_out_s[i] = p_act_input_s[i] / out;
    }
}
}
}  // namespace shave_lib
}  // namespace nn
