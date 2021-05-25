//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include <sstream>
#include <vector>

#include "ie_common.h"

namespace vpu {

namespace KmbPlugin {

namespace utils {

struct priorBoxParam {
    priorBoxParam(float offset, float step, const std::vector<float>& min_sizes, const std::vector<float>& max_sizes,
                  bool flip, bool clip, bool scale_all_sizes, const std::vector<float>& fixed_sizes,
                  const std::vector<float>& fixed_ratios, const std::vector<float>& densitys,
                  const std::vector<float>& src_aspect_ratios, const std::vector<float>& src_variance,
                  const InferenceEngine::SizeVector& data_dims, const InferenceEngine::SizeVector& image_dims,
                  const InferenceEngine::SizeVector& out_dims)
            : _offset(offset),
              _step(step),
              _min_sizes(min_sizes),
              _max_sizes(max_sizes),
              _flip(flip),
              _clip(clip),
              _scale_all_sizes(scale_all_sizes),
              _fixed_sizes(fixed_sizes),
              _fixed_ratios(fixed_ratios),
              _densitys(densitys),
              _src_aspect_ratios(src_aspect_ratios),
              _src_variance(src_variance),
              _data_dims(data_dims),
              _image_dims(image_dims),
              _out_dims(out_dims) {
    }

    float _offset;
    float _step;
    std::vector<float> _min_sizes;
    std::vector<float> _max_sizes;
    bool _flip;
    bool _clip;
    bool _scale_all_sizes;
    std::vector<float> _fixed_sizes;
    std::vector<float> _fixed_ratios;
    std::vector<float> _densitys;
    std::vector<float> _src_aspect_ratios;
    std::vector<float> _src_variance;
    InferenceEngine::SizeVector _data_dims;
    InferenceEngine::SizeVector _image_dims;
    InferenceEngine::SizeVector _out_dims;
};
struct priorBoxClusteredParam {
    float _offset;
    int _clip;
    float _step_w;
    float _step_h;
    int _layer_width;
    int _layer_height;
    int _img_width;
    int _img_height;
    int _num_priors;
    std::vector<float> _widths;
    std::vector<float> _heights;
    std::vector<float> _variance;
    int _size;
};

std::vector<double> computePriorbox(const priorBoxParam& param);
std::vector<double> computePriorboxClustered(const priorBoxClusteredParam& param);

template <typename T>
std::string vectorToStr(const std::vector<T>& array) {
    std::stringstream outStream;

    for (size_t i = 0; i < array.size(); ++i) {
        if (i != 0) {
            outStream << ',';
        }
        outStream << array[i];
    }
    return outStream.str();
}

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
