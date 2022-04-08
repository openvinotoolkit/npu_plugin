//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorConvert.h>
#include <moviVectorTypes.h>
#include <mv_types.h>
#include <param_region_yolo.h>

#define VECTOR_SIZE (8)     /* Changes to this should be reflected in the code as well */
#define ND_NCHW_REV 0x4321  // reverse code of ND_NCHW 0x1234
#define ND_NHWC_REV 0x2431  // reverse code of ND_NHWC 0x1342
#define ND_CHW_REV 0x321    // reverse code of ND_CHW 0x123
#define ND_HWC_REV 0x132    // reverse code of ND_HWC 0x231

#define intrinsic_vau(intrinsic, vin, vout) (vout) = intrinsic((vin))
#define intrinsic_sau(intrinsic, in, out) (out) = intrinsic((in))
#define intrinsic_cmu(intrinsic, in, out) (out) = intrinsic(in, out)

#define vau_exp(vin, vout) (intrinsic_vau(__builtin_shave_vau_exp_v8f16_r, vin, vout))

// Compute exp(x) ≈ 2 ^ (x * 0x3dc5)
#define sau_exp(in, out)                                                     \
    {                                                                        \
        const uint16_t inv_ln2 = 0x3dc5;                                     \
        const half inv_ln2_h = *(const half*)&inv_ln2;                       \
        intrinsic_sau(__builtin_shave_sau_exp2_f16_l_r, in* inv_ln2_h, out); \
    }

#define cmu_max_v(vin, vout) (intrinsic_cmu(__builtin_shave_cmu_max_f16_rr_half8, vin, vout))
#define cmu_max_s(in, out) (intrinsic_cmu(__builtin_shave_cmu_max_f16_rr_half, in, out))

using namespace sw_params;

namespace {

void sigmoid_calculate_NCHW(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                            int32_t end_index, int32_t batch, int32_t channel, int32_t height, int32_t width) {
    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t reg = 0; reg < regions; ++reg) {
            int32_t index_offset = n * channel * height * width + reg * (coords + 1 + classes) * height * width;
            half* in = tmp_in_s + index_offset;
            half* out = tmp_out_s + index_offset;
            int32_t n_ele_pline = 2 * height * width;
            for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                half8 val;
                vau_exp(((-1.0f) * (((half8*)in)[i])), val);
                ((half8*)out)[i] = 1.0f / (1.0f + val);
            }
            for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                half val;
                sau_exp(((-1.0f) * (in[i])), val);
                out[i] = 1.0f / (1.0f + val);
            }
            in = in + coords * height * width;
            out = out + coords * height * width;
            n_ele_pline = end_index * height * width;
            for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                half8 val;
                vau_exp(((-1.0f) * (((half8*)in)[i])), val);
                ((half8*)out)[i] = 1.0f / (1.0f + val);
            }
            for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                half val;
                sau_exp(((-1.0f) * (in[i])), val);
                out[i] = 1.0f / (1.0f + val);
            }
        }
    }
}

void sigmoid_calculate_NHWC(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                            int32_t end_index, int32_t batch, int32_t height, int32_t width, int32_t channel) {
    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++w) {
                for (int32_t reg = 0; reg < regions; ++reg) {
                    for (int32_t chan = 0; chan < 2; ++chan) {
                        int32_t index = n * height * width * channel + (h * width + w) * channel +
                                        reg * (coords + 1 + classes) + chan;
                        half val;
                        sau_exp(((-1.0f) * tmp_in_s[index]), val);
                        tmp_out_s[index] = 1.0f / (1.0f + val);
                    }

                    for (int32_t chan = coords; chan < coords + end_index; ++chan) {
                        int32_t index = n * height * width * channel + (h * width + w) * channel +
                                        reg * (coords + 1 + classes) + chan;
                        half val;
                        sau_exp(((-1.0f) * tmp_in_s[index]), val);
                        tmp_out_s[index] = 1.0f / (1.0f + val);
                    }
                }
            }
        }
    }
}

void sigmoid_calculate_CHW(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                           int32_t end_index, int32_t height, int32_t width) {
    for (int32_t reg = 0; reg < regions; ++reg) {
        int32_t index_offset = reg * (coords + 1 + classes) * height * width;
        half* in = tmp_in_s + index_offset;
        half* out = tmp_out_s + index_offset;
        int32_t n_ele_pline = 2 * height * width;
        for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
            half8 val;
            vau_exp(((-1.0f) * (((half8*)in)[i])), val);
            ((half8*)out)[i] = 1.0f / (1.0f + val);
        }
        for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
            half val;
            sau_exp(((-1.0f) * (in[i])), val);
            out[i] = 1.0f / (1.0f + val);
        }
        in = in + coords * height * width;
        out = out + coords * height * width;
        n_ele_pline = end_index * height * width;
        for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
            half8 val;
            vau_exp(((-1.0f) * (((half8*)in)[i])), val);
            ((half8*)out)[i] = 1.0f / (1.0f + val);
        }
        for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
            half val;
            sau_exp(((-1.0f) * (in[i])), val);
            out[i] = 1.0f / (1.0f + val);
        }
    }
}

void sigmoid_calculate_HWC(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                           int32_t end_index, int32_t height, int32_t width) {
    for (int32_t h = 0; h < height; ++h) {
        for (int32_t w = 0; w < width; ++w) {
            for (int32_t reg = 0; reg < regions; ++reg) {
                for (int32_t chan = 0; chan < 2; ++chan) {
                    int32_t index =
                            (h * width + w) * regions * (coords + 1 + classes) + reg * (coords + 1 + classes) + chan;
                    half val;
                    sau_exp(((-1.0f) * tmp_in_s[index]), val);
                    tmp_out_s[index] = 1.0f / (1.0f + val);
                }

                for (int32_t chan = coords; chan < coords + end_index; ++chan) {
                    int32_t index =
                            (h * width + w) * regions * (coords + 1 + classes) + reg * (coords + 1 + classes) + chan;
                    half val;
                    sau_exp(((-1.0f) * tmp_in_s[index]), val);
                    tmp_out_s[index] = 1.0f / (1.0f + val);
                }
            }
        }
    }
}

void softmax_calculate_NHWC(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                            int32_t batch, int32_t height, int32_t width, int32_t channel) {
    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++w) {
                for (int32_t reg = 0; reg < regions; ++reg) {
                    int32_t index_offset = n * height * width * channel +
                                           regions * (coords + 1 + classes) * (h * width + w) +
                                           reg * (coords + 1 + classes) + (coords + 1);
                    half max = tmp_in_s[index_offset];
                    for (int32_t cla = 0; cla < classes; ++cla) {
                        cmu_max_s(tmp_in_s[index_offset + cla], max);
                    }
                    half sum = 0;
                    for (int32_t cla = 0; cla < classes; ++cla) {
                        half out;
                        sau_exp((tmp_in_s[index_offset + cla] - max), out);
                        tmp_out_s[index_offset + cla] = out;
                        sum += tmp_out_s[index_offset + cla];
                    }
                    for (int32_t cla = 0; cla < classes; ++cla) {
                        tmp_out_s[index_offset + cla] /= sum;
                    }
                }
            }
        }
    }
}

void softmax_calculate_NCHW(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                            int32_t batch, int32_t channels, int32_t height, int32_t width) {
    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t reg = 0; reg < regions; ++reg) {
            int32_t index_offset = n * channels * height * width + reg * (coords + 1 + classes) * height * width +
                                   (coords + 1) * height * width;
            half* in = tmp_in_s + index_offset;
            half* out = tmp_out_s + index_offset;
            int32_t n_ele_pline = height * width;
            half max[n_ele_pline];
            for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                ((half8*)max)[i] = ((half8*)in)[i];
            }
            for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                max[i] = in[i];
            }
            for (int32_t cla = 0; cla < classes; ++cla) {
                for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                    cmu_max_v(((half8*)(in + cla * height * width))[i], ((half8*)max)[i]);
                }
                for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                    cmu_max_s((in + cla * height * width)[i], max[i]);
                }
            }
            half sum[n_ele_pline];
            for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                ((half8*)sum)[i] = 0;
            }
            for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                sum[i] = 0;
            }
            for (int32_t cla = 0; cla < classes; ++cla) {
                for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                    half8 val;
                    vau_exp((((half8*)(in + cla * height * width))[i] - ((half8*)max)[i]), val);
                    ((half8*)(out + cla * height * width))[i] = val;
                    ((half8*)sum)[i] += val;
                }
                for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                    half val;
                    sau_exp(((in + cla * height * width)[i] - max[i]), val);
                    (out + cla * height * width)[i] = val;
                    sum[i] += val;
                }
            }
            for (int32_t cla = 0; cla < classes; ++cla) {
                for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                    ((half8*)(out + cla * height * width))[i] /= ((half8*)sum)[i];
                }
                for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                    (out + cla * height * width)[i] /= sum[i];
                }
            }
        }
    }
}

void softmax_calculate_HWC(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                           int32_t height, int32_t width) {
    for (int32_t h = 0; h < height; ++h) {
        for (int32_t w = 0; w < width; ++w) {
            for (int32_t reg = 0; reg < regions; ++reg) {
                int32_t index_offset =
                        regions * (coords + 1 + classes) * (h * width + w) + reg * (coords + 1 + classes) + coords + 1;
                half max = tmp_in_s[index_offset];
                for (int32_t cla = 0; cla < classes; ++cla) {
                    cmu_max_s(tmp_in_s[index_offset + cla], max);
                }
                half sum = 0;
                for (int32_t cla = 0; cla < classes; ++cla) {
                    half out;
                    sau_exp((tmp_in_s[index_offset + cla] - max), out);
                    tmp_out_s[index_offset + cla] = out;
                    sum += tmp_out_s[index_offset + cla];
                }
                for (int32_t cla = 0; cla < classes; ++cla) {
                    tmp_out_s[index_offset + cla] /= sum;
                }
            }
        }
    }
}

void softmax_calculate_CHW(half* tmp_in_s, half* tmp_out_s, int64_t coords, int64_t regions, int64_t classes,
                           int32_t height, int32_t width) {
    for (int32_t reg = 0; reg < regions; ++reg) {
        int32_t index_offset = reg * (coords + 1 + classes) * height * width + (coords + 1) * height * width;
        half* in = tmp_in_s + index_offset;
        half* out = tmp_out_s + index_offset;
        int32_t n_ele_pline = height * width;
        half max[n_ele_pline];
        for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
            ((half8*)max)[i] = ((half8*)in)[i];
        }
        for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
            max[i] = in[i];
        }
        for (int32_t cla = 0; cla < classes; ++cla) {
            for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                cmu_max_v(((half8*)(in + cla * height * width))[i], ((half8*)max)[i]);
            }
            for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                cmu_max_s((in + cla * height * width)[i], max[i]);
            }
        }
        half sum[n_ele_pline];
        for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
            ((half8*)sum)[i] = 0;
        }
        for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
            sum[i] = 0;
        }
        for (int32_t cla = 0; cla < classes; ++cla) {
            for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                half8 val;
                vau_exp((((half8*)(in + cla * height * width))[i] - ((half8*)max)[i]), val);
                ((half8*)(out + cla * height * width))[i] = val;
                ((half8*)sum)[i] += val;
            }
            for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                half val;
                sau_exp(((in + cla * height * width)[i] - max[i]), val);
                (out + cla * height * width)[i] = val;
                sum[i] += val;
            }
        }
        for (int32_t cla = 0; cla < classes; ++cla) {
            for (int32_t i = 0; i < n_ele_pline / VECTOR_SIZE; ++i) {
                ((half8*)(out + cla * height * width))[i] /= ((half8*)sum)[i];
            }
            for (int32_t i = (n_ele_pline / VECTOR_SIZE) * VECTOR_SIZE; i < n_ele_pline; ++i) {
                (out + cla * height * width)[i] /= sum[i];
            }
        }
    }
}

}  // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void single_shave_region_yolo(const struct RegionYoloParams* lParams) {
    half8* p_act_data_v = (half8*)(lParams->input.dataAddr);  // 0x1F000000
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);  // 0x1F004000

    half* p_act_data_s = (half*)(lParams->input.dataAddr);  // 0x1F000000
    half* p_act_out_s = (half*)(lParams->output.dataAddr);  // 0x1F004000

    int32_t* inputDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t* outputDims = (int32_t*)(lParams->output.dimsAddr);
    int32_t numInputDims = (int32_t)(lParams->input.numDims);
    int32_t numOutputDims = (int32_t)(lParams->output.numDims);

    int64_t coords = (int64_t)(lParams->coords);
    int64_t classes = (int64_t)(lParams->classes);
    int64_t regions = (int64_t)(lParams->regions);
    int64_t mask_size = (int64_t)(lParams->mask_size);
    uint64_t do_softmax = (uint64_t)(lParams->do_softmax);
    uint64_t order = (uint64_t)(lParams->input.dimsOrder);

    int32_t numInputElements = 1;

    for (int32_t i = 0; i != numInputDims; i++) {
        numInputElements *= inputDims[i];
    }
    const int numVectors = numInputElements / VECTOR_SIZE;

    int32_t end_index;

    if (do_softmax) {  // Yolo V2
        end_index = 1;
    } else {  // Yolo V3
        regions = mask_size;
        end_index = 1 + classes;
    }

#pragma clang loop unroll_count(8)
    for (int32_t i = 0; i < numVectors; i++) {
        p_act_out_v[i] = p_act_data_v[i];
    }
    for (int32_t i = numVectors * VECTOR_SIZE; i < numInputElements; i++) {
        p_act_out_s[i] = p_act_data_s[i];
    }

    if (order == ND_NCHW || order == ND_NCHW_REV) {  // NCHW  0x1234
        sigmoid_calculate_NCHW(p_act_out_s, p_act_out_s, coords, regions, classes, end_index, inputDims[3],
                               inputDims[2], inputDims[1], inputDims[0]);
    } else if (order == ND_CHW || order == ND_CHW_REV) {  // CHW  0x123
        sigmoid_calculate_CHW(p_act_out_s, p_act_out_s, coords, regions, classes, end_index, inputDims[1],
                              inputDims[0]);
    } else if (order == ND_NHWC || order == ND_NHWC_REV) {  // NHWC  0x1342
        sigmoid_calculate_NHWC(p_act_out_s, p_act_out_s, coords, regions, classes, end_index, inputDims[3],
                               inputDims[2], inputDims[1], inputDims[0]);
    } else {  // HWC 0x231
        sigmoid_calculate_HWC(p_act_out_s, p_act_out_s, coords, regions, classes, end_index, inputDims[2],
                              inputDims[1]);
    }
    if (do_softmax) {
        if (order == ND_NCHW || order == ND_NCHW_REV) {  // NCHW
            softmax_calculate_NCHW(p_act_out_s, p_act_out_s, coords, regions, classes, inputDims[3], inputDims[2],
                                   inputDims[1], inputDims[0]);
        } else if (order == ND_CHW || order == ND_CHW_REV) {  // CHW
            softmax_calculate_CHW(p_act_out_s, p_act_out_s, coords, regions, classes, inputDims[1], inputDims[0]);
        } else if (order == ND_NHWC || order == ND_NHWC_REV) {  // NHWC
            softmax_calculate_NHWC(p_act_out_s, p_act_out_s, coords, regions, classes, inputDims[3], inputDims[2],
                                   inputDims[1], inputDims[0]);
        } else {  // HWC
            softmax_calculate_HWC(p_act_out_s, p_act_out_s, coords, regions, classes, inputDims[2], inputDims[1]);
        }
    }
}
}
}  // namespace shave_lib
}  // namespace nn
