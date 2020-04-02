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

std::vector<double> computePriorbox(const InferenceEngine::CNNLayerPtr& layer) {
    if (layer->insData.size() != 2 || layer->outData.empty())
        THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

    if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4 ||
        layer->insData[1].lock()->getTensorDesc().getDims().size() != 4)
        THROW_IE_EXCEPTION << "PriorBox supports only 4D blobs!";

    // parse settings for priorbox layer
    float offset = layer->GetParamAsFloat("offset");
    float step = layer->GetParamAsFloat("step", 0.f);
    std::vector<float> min_sizes = layer->GetParamAsFloats("min_size", {});
    std::vector<float> max_sizes = layer->GetParamAsFloats("max_size", {});
    bool flip = layer->GetParamAsBool("flip", false);
    bool clip = layer->GetParamAsBool("clip", false);
    bool scale_all_sizes = layer->GetParamAsBool("scale_all_sizes", true);

    std::vector<float> fixed_sizes = layer->GetParamAsFloats("fixed_size", {});
    std::vector<float> fixed_ratios = layer->GetParamAsFloats("fixed_ratio", {});
    std::vector<float> densitys = layer->GetParamAsFloats("density", {});

    const std::vector<float> src_aspect_ratios = layer->GetParamAsFloats("aspect_ratio", {});

    std::vector<float> dst_aspect_ratios {1.0f};

    bool exist = false;
    for (float src_aspect_ratio : src_aspect_ratios) {
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

        if (flip) {
            dst_aspect_ratios.push_back(1.0f / src_aspect_ratio);
        }
    }

    int num_priors = 0;

    if (scale_all_sizes) {
        num_priors = static_cast<int>(dst_aspect_ratios.size() * min_sizes.size());
    } else {
        num_priors = static_cast<int>(dst_aspect_ratios.size() + min_sizes.size() - 1);
    }

    if (fixed_sizes.size() > 0) {
        num_priors = static_cast<int>(dst_aspect_ratios.size() * fixed_sizes.size());
    }

    if (densitys.size() > 0) {
        for (size_t i = 0; i < densitys.size(); ++i) {
            if (fixed_ratios.size() > 0) {
                num_priors += (fixed_ratios.size()) * (static_cast<size_t>(pow(densitys[i], 2)) - 1);
            } else {
                num_priors += (dst_aspect_ratios.size()) * (static_cast<size_t>(pow(densitys[i], 2)) - 1);
            }
        }
    }

    for (auto it = max_sizes.begin(); it != max_sizes.end(); it++) {
        num_priors += 1;
    }

    const std::vector<float> src_variance = layer->GetParamAsFloats("variance", {});

    std::vector<float> dst_variance;

    if (src_variance.size() == 1 || src_variance.size() == 4) {
        for (float i : src_variance) {
            if (i < 0) {
                THROW_IE_EXCEPTION << "Variance must be > 0.";
            }

            dst_variance.push_back(i);
        }
    } else if (src_variance.empty()) {
        dst_variance.push_back(0.1f);
    } else {
        THROW_IE_EXCEPTION << "Wrong number of variance values. Not less than 1 and more than 4 variance values.";
    }

    auto& dataMemPtr = layer->insData[0];
    auto& imageMemPtr = layer->insData[1];
    auto& dstMemPtr = layer->outData[0];
    SizeVector _data_dims = dataMemPtr.lock()->getTensorDesc().getDims();
    SizeVector _image_dims = imageMemPtr.lock()->getTensorDesc().getDims();

    const int W = _data_dims[3];
    const int H = _data_dims[2];
    const int IW = _image_dims[3];
    const int IH = _image_dims[2];

    int layer_width = layer->insData[0].lock()->getTensorDesc().getDims()[3];
    int layer_height = layer->insData[0].lock()->getTensorDesc().getDims()[2];

    int img_width = layer->insData[1].lock()->getTensorDesc().getDims()[3];
    int img_height = layer->insData[1].lock()->getTensorDesc().getDims()[2];

    const int OH = dstMemPtr->getTensorDesc().getDims()[2];
    const int OW = (dstMemPtr->getTensorDesc().getDims().size() == 3) ? 1 : dstMemPtr->getTensorDesc().getDims()[3];

    float step_x = 0.0f;
    float step_y = 0.0f;

    if (step == 0) {
        step_x = static_cast<float>(IW) / W;
        step_y = static_cast<float>(IH) / H;
    } else {
        step_x = step;
        step_y = step;
    }

    float IWI = 1.0f / static_cast<float>(IW);
    float IHI = 1.0f / static_cast<float>(IH);

    auto dims = layer->outData.front()->getDims();

    std::vector<double> out_boxes(dims[1] * dims[2], 0.);
    double* dst_data = out_boxes.data();

    int idx = 0;
    float center_x = 0.0f;
    float center_y = 0.0f;

    float box_width;
    float box_height;

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            if (step == 0) {
                center_x = (w + 0.5f) * step_x;
                center_y = (h + 0.5f) * step_y;
            } else {
                center_x = (offset + w) * step;
                center_y = (offset + h) * step;
            }

            for (size_t s = 0; s < fixed_sizes.size(); ++s) {
                size_t fixed_size_ = static_cast<size_t>(fixed_sizes[s]);
                box_width = box_height = fixed_size_ * 0.5f;

                if (fixed_ratios.size() > 0) {
                    for (float ar : fixed_ratios) {
                        size_t density_ = static_cast<size_t>(densitys[s]);
                        int shift = static_cast<int>(fixed_sizes[s] / density_);
                        ar = sqrt(ar);
                        float box_width_ratio = fixed_sizes[s] * 0.5f * ar;
                        float box_height_ratio = fixed_sizes[s] * 0.5f / ar;
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
                    if (densitys.size() > 0) {
                        int density_ = static_cast<int>(densitys[s]);
                        int shift = static_cast<int>(fixed_sizes[s] / density_);
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

                        int density_ = static_cast<int>(densitys[s]);
                        int shift = static_cast<int>(fixed_sizes[s] / density_);
                        ar = sqrt(ar);
                        float box_width_ratio = fixed_sizes[s] * 0.5f * ar;
                        float box_height_ratio = fixed_sizes[s] * 0.5f / ar;
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

            for (size_t msIdx = 0; msIdx < min_sizes.size(); msIdx++) {
                box_width = min_sizes[msIdx] * 0.5f;
                box_height = min_sizes[msIdx] * 0.5f;

                dst_data[idx++] = (center_x - box_width) * IWI;
                dst_data[idx++] = (center_y - box_height) * IHI;
                dst_data[idx++] = (center_x + box_width) * IWI;
                dst_data[idx++] = (center_y + box_height) * IHI;

                if (max_sizes.size() > msIdx) {
                    box_width = box_height = sqrt(min_sizes[msIdx] * max_sizes[msIdx]) * 0.5f;

                    dst_data[idx++] = (center_x - box_width) * IWI;
                    dst_data[idx++] = (center_y - box_height) * IHI;
                    dst_data[idx++] = (center_x + box_width) * IWI;
                    dst_data[idx++] = (center_y + box_height) * IHI;
                }

                if (scale_all_sizes || (!scale_all_sizes && (msIdx == min_sizes.size() - 1))) {
                    size_t sIdx = scale_all_sizes ? msIdx : 0;
                    for (float ar : dst_aspect_ratios) {
                        if (fabs(ar - 1.0f) < 1e-6) {
                            continue;
                        }

                        ar = sqrt(ar);
                        box_width = min_sizes[sIdx] * 0.5f * ar;
                        box_height = min_sizes[sIdx] * 0.5f / ar;

                        dst_data[idx++] = (center_x - box_width) * IWI;
                        dst_data[idx++] = (center_y - box_height) * IHI;
                        dst_data[idx++] = (center_x + box_width) * IWI;
                        dst_data[idx++] = (center_y + box_height) * IHI;
                    }
                }
            }
        }
    }

    if (clip) {
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

}  // namespace ParseLayersHelpers
}  // namespace vpu
