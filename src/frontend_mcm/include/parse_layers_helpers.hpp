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

#pragma once

#include <sstream>
#include <vector>

#include "ie_common.h"

namespace vpu {
namespace ParseLayersHelpers {

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
          _out_dims(out_dims) {}

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
    if (array.empty()) return std::string();

    std::stringstream outStream;

    for (size_t i = 0; i < array.size(); ++i) {
        if (i != 0) {
            outStream << ',';
        }
        outStream << array[i];
    }
    return outStream.str();
}

}  // namespace ParseLayersHelpers
}  // namespace vpu
