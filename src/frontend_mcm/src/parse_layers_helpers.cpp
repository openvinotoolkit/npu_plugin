//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "parse_layers_helpers.hpp"

#include <cmath>

#include "ie_layers.h"
#include "ie_parallel.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace vpu {
namespace ParseLayersHelpers {

namespace {

static inline float clip_great(float x, float threshold) { return x < threshold ? x : threshold; }

static inline float clip_less(float x, float threshold) { return x > threshold ? x : threshold; }

}  // namespace

std::vector<double> computePriorbox(const priorBoxParam& param) {
    std::vector<float> dst_aspect_ratios {1.0f};

    bool exist = false;
    for (float src_aspect_ratio : param._src_aspect_ratios) {
        exist = false;

        if (std::fabs(src_aspect_ratio) < std::numeric_limits<float>::epsilon()) {
            THROW_IE_EXCEPTION << "aspect_ratio param can't be equal to zero";
        }

        // skip existing ratios
        for (float dst_aspect_ratio : dst_aspect_ratios) {
            if (fabs(src_aspect_ratio - dst_aspect_ratio) < 1e-6) {
                exist = true;
                break;
            }
        }

        if (exist) {
            continue;
        }

        dst_aspect_ratios.push_back(src_aspect_ratio);

        if (param._flip) {
            dst_aspect_ratios.push_back(1.0f / src_aspect_ratio);
        }
    }

    int num_priors = 0;

    if (param._scale_all_sizes) {
        num_priors = static_cast<int>(dst_aspect_ratios.size() * param._min_sizes.size());
    } else {
        num_priors = static_cast<int>(dst_aspect_ratios.size() + param._min_sizes.size() - 1);
    }

    if (param._fixed_sizes.size() > 0) {
        num_priors = static_cast<int>(dst_aspect_ratios.size() * param._fixed_sizes.size());
    }

    if (param._densitys.size() > 0) {
        for (size_t i = 0; i < param._densitys.size(); ++i) {
            if (param._fixed_ratios.size() > 0) {
                num_priors += (param._fixed_ratios.size()) * (static_cast<size_t>(pow(param._densitys[i], 2)) - 1);
            } else {
                num_priors += (dst_aspect_ratios.size()) * (static_cast<size_t>(pow(param._densitys[i], 2)) - 1);
            }
        }
    }

    for (auto it = param._max_sizes.begin(); it != param._max_sizes.end(); it++) {
        num_priors += 1;
    }

    std::vector<float> dst_variance;

    if (param._src_variance.size() == 1 || param._src_variance.size() == 4) {
        for (float i : param._src_variance) {
            if (i < 0) {
                THROW_IE_EXCEPTION << "Variance must be > 0.";
            }

            dst_variance.push_back(i);
        }
    } else if (param._src_variance.empty()) {
        dst_variance.push_back(0.1f);
    } else {
        THROW_IE_EXCEPTION << "Wrong number of variance values. Not less than 1 and more than 4 variance values.";
    }

    const size_t W = param._data_dims[3];
    const size_t H = param._data_dims[2];
    const size_t IW = param._image_dims[3];
    const size_t IH = param._image_dims[2];

    const size_t OH = param._out_dims[2];
    const size_t OW = (param._out_dims.size() == 3) ? 1 : param._out_dims[3];

    float step_x = 0.0f;
    float step_y = 0.0f;

    if (param._step == 0) {
        step_x = static_cast<float>(IW) / W;
        step_y = static_cast<float>(IH) / H;
    } else {
        step_x = param._step;
        step_y = param._step;
    }

    float IWI = 1.0f / static_cast<float>(IW);
    float IHI = 1.0f / static_cast<float>(IH);

    size_t out_size = 1;
    for (size_t i = 0; i < param._out_dims.size(); ++i) {
        out_size *= param._out_dims[i];
    }
    std::vector<double> out_boxes(out_size, 0.);
    double* dst_data = out_boxes.data();

    int idx = 0;
    float center_x = 0.0f;
    float center_y = 0.0f;

    float box_width;
    float box_height;

    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            if (param._step == 0) {
                center_x = (w + 0.5f) * step_x;
                center_y = (h + 0.5f) * step_y;
            } else {
                center_x = (param._offset + w) * param._step;
                center_y = (param._offset + h) * param._step;
            }

            for (size_t s = 0; s < param._fixed_sizes.size(); ++s) {
                size_t fixed_size_ = static_cast<size_t>(param._fixed_sizes[s]);
                box_width = box_height = fixed_size_ * 0.5f;

                if (param._fixed_ratios.size() > 0) {
                    for (float ar : param._fixed_ratios) {
                        size_t density_ = static_cast<size_t>(param._densitys[s]);
                        int shift = static_cast<int>(param._fixed_sizes[s] / density_);
                        ar = sqrt(ar);
                        float box_width_ratio = param._fixed_sizes[s] * 0.5f * ar;
                        float box_height_ratio = param._fixed_sizes[s] * 0.5f / ar;
                        for (size_t r = 0; r < density_; ++r) {
                            for (size_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                // xmin
                                dst_data[idx++] = clip_less((center_x_temp - box_width_ratio) * IWI, 0);
                                // ymin
                                dst_data[idx++] = clip_less((center_y_temp - box_height_ratio) * IHI, 0);
                                // xmax
                                dst_data[idx++] = clip_great((center_x_temp + box_width_ratio) * IWI, 1);
                                // ymax
                                dst_data[idx++] = clip_great((center_y_temp + box_height_ratio) * IHI, 1);
                            }
                        }
                    }
                } else {
                    if (param._densitys.size() > 0) {
                        int density_ = static_cast<int>(param._densitys[s]);
                        int shift = static_cast<int>(param._fixed_sizes[s] / density_);
                        for (int r = 0; r < density_; ++r) {
                            for (int c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                // xmin
                                dst_data[idx++] = clip_less((center_x_temp - box_width) * IWI, 0);
                                // ymin
                                dst_data[idx++] = clip_less((center_y_temp - box_height) * IHI, 0);
                                // xmax
                                dst_data[idx++] = clip_great((center_x_temp + box_width) * IWI, 1);
                                // ymax
                                dst_data[idx++] = clip_great((center_y_temp + box_height) * IHI, 1);
                            }
                        }
                    }
                    //  Rest of priors
                    for (float ar : dst_aspect_ratios) {
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        int density_ = static_cast<int>(param._densitys[s]);
                        int shift = static_cast<int>(param._fixed_sizes[s] / density_);
                        ar = sqrt(ar);
                        float box_width_ratio = param._fixed_sizes[s] * 0.5f * ar;
                        float box_height_ratio = param._fixed_sizes[s] * 0.5f / ar;
                        for (int r = 0; r < density_; ++r) {
                            for (int c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                                // xmin
                                dst_data[idx++] = clip_less((center_x_temp - box_width_ratio) * IWI, 0);
                                // ymin
                                dst_data[idx++] = clip_less((center_y_temp - box_height_ratio) * IHI, 0);
                                // xmax
                                dst_data[idx++] = clip_great((center_x_temp + box_width_ratio) * IWI, 1);
                                // ymax
                                dst_data[idx++] = clip_great((center_y_temp + box_height_ratio) * IHI, 1);
                            }
                        }
                    }
                }
            }

            for (size_t msIdx = 0; msIdx < param._min_sizes.size(); msIdx++) {
                box_width = param._min_sizes[msIdx] * 0.5f;
                box_height = param._min_sizes[msIdx] * 0.5f;

                dst_data[idx++] = (center_x - box_width) * IWI;
                dst_data[idx++] = (center_y - box_height) * IHI;
                dst_data[idx++] = (center_x + box_width) * IWI;
                dst_data[idx++] = (center_y + box_height) * IHI;

                if (param._max_sizes.size() > msIdx) {
                    box_width = box_height = sqrt(param._min_sizes[msIdx] * param._max_sizes[msIdx]) * 0.5f;

                    dst_data[idx++] = (center_x - box_width) * IWI;
                    dst_data[idx++] = (center_y - box_height) * IHI;
                    dst_data[idx++] = (center_x + box_width) * IWI;
                    dst_data[idx++] = (center_y + box_height) * IHI;
                }

                if (param._scale_all_sizes || (!param._scale_all_sizes && (msIdx == param._min_sizes.size() - 1))) {
                    size_t sIdx = param._scale_all_sizes ? msIdx : 0;
                    for (float ar : dst_aspect_ratios) {
                        if (fabs(ar - 1.0f) < 1e-6) {
                            continue;
                        }

                        ar = sqrt(ar);
                        box_width = param._min_sizes[sIdx] * 0.5f * ar;
                        box_height = param._min_sizes[sIdx] * 0.5f / ar;

                        dst_data[idx++] = (center_x - box_width) * IWI;
                        dst_data[idx++] = (center_y - box_height) * IHI;
                        dst_data[idx++] = (center_x + box_width) * IWI;
                        dst_data[idx++] = (center_y + box_height) * IHI;
                    }
                }
            }
        }
    }

    if (param._clip) {
        parallel_for((H * W * num_priors * 4), [&](size_t i) {
            dst_data[i] = std::min(std::max(static_cast<float>(dst_data[i]), 0.0f), 1.0f);
        });
    }

    size_t channel_size = OH * OW;
    dst_data += channel_size;
    if (dst_variance.size() == 1) {
        parallel_for(channel_size, [&](size_t i) {
            dst_data[i] = dst_variance[0];
        });
    } else {
        parallel_for((H * W * num_priors), [&](size_t i) {
            for (size_t j = 0; j < 4; ++j) {
                dst_data[i * 4 + j] = dst_variance[j];
            }
        });
    }
    return out_boxes;
}

std::vector<double> computePriorboxClustered(const priorBoxClusteredParam& param) {
    std::vector<double> boxes(param._size);
    for (int h = 0; h < param._layer_height; ++h) {
        for (int w = 0; w < param._layer_width; ++w) {
            float center_x = (w + param._offset) * param._step_w;
            float center_y = (h + param._offset) * param._step_h;
            int plane_shift = param._size / 2;
            int var_size = param._variance.size();

            for (int s = 0; s < param._num_priors; ++s) {
                float box_width = param._widths[s];
                float box_height = param._heights[s];

                float xmin = (center_x - box_width / 2.0f) / param._img_width;
                float ymin = (center_y - box_height / 2.0f) / param._img_height;
                float xmax = (center_x + box_width / 2.0f) / param._img_width;
                float ymax = (center_y + box_height / 2.0f) / param._img_height;

                if (param._clip) {
                    xmin = (std::min)((std::max)(xmin, 0.0f), 1.0f);
                    ymin = (std::min)((std::max)(ymin, 0.0f), 1.0f);
                    xmax = (std::min)((std::max)(xmax, 0.0f), 1.0f);
                    ymax = (std::min)((std::max)(ymax, 0.0f), 1.0f);
                }

                boxes[h * param._layer_width * param._num_priors * 4 + w * param._num_priors * 4 + s * 4 + 0] = xmin;
                boxes[h * param._layer_width * param._num_priors * 4 + w * param._num_priors * 4 + s * 4 + 1] = ymin;
                boxes[h * param._layer_width * param._num_priors * 4 + w * param._num_priors * 4 + s * 4 + 2] = xmax;
                boxes[h * param._layer_width * param._num_priors * 4 + w * param._num_priors * 4 + s * 4 + 3] = ymax;

                for (int j = 0; j < var_size; j++) {
                    boxes[plane_shift + h * param._layer_width * param._num_priors * var_size +
                          w * param._num_priors * var_size + s * var_size + j] = param._variance[j];
                }
            }
        }
    }
    return boxes;
}

}  // namespace ParseLayersHelpers
}  // namespace vpu
