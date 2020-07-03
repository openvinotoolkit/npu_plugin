//
// Copyright 2019 Intel Corporation.
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

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct PSROIPoolingParams final {
    size_t output_dim_    = 0u;
    size_t group_size_    = 0u;
    size_t spatial_bin_x_ = 1u;
    size_t spatial_bin_y_ = 1u;
    float spatial_scale_  = 0.0f;
    std::string mode_;

    PSROIPoolingParams& output_dim(size_t output_dim) {
        output_dim_ = output_dim;
        return *this;
    }

    PSROIPoolingParams& group_size(size_t group_size) {
        group_size_ = group_size;
        return *this;
    }

    PSROIPoolingParams& spatial_bin_x(size_t spatial_bin_x) {
        spatial_bin_x_ = spatial_bin_x;
        return *this;
    }

    PSROIPoolingParams& spatial_bin_y(size_t spatial_bin_y) {
        spatial_bin_y_ = spatial_bin_y;
        return *this;
    }

    PSROIPoolingParams& spatial_scale(float spatial_scale) {
        spatial_scale_ = spatial_scale;
        return *this;
    }

    PSROIPoolingParams& mode(std::string mode) {
        mode_ = std::move(mode);
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const PSROIPoolingParams& p) {
        vpu::formatPrint(os, "[output_dim:%v, group_size:%v, spatial_bin_x:%v, spatial_bin_y:%v, spatial_scale:%v, mode:%v]",
                p.output_dim_, p.group_size_, p.spatial_bin_x_, p.spatial_bin_y_, p.spatial_scale_, p.mode_);
	return os;
}

struct PSROIPoolingLayerDef final {
    TestNetwork& net_;
    std::string name_;
    PSROIPoolingParams params_;

    PortInfo input_port_;
    PortInfo coords_port_;

    PSROIPoolingLayerDef(TestNetwork& net, std::string name, PSROIPoolingParams params)
        : net_(net), name_(std::move(name)), params_(std::move(params)) {
    }

    PSROIPoolingLayerDef& input(const std::string& name, size_t index = 0) {
        input_port_ = PortInfo(name, index);
        return *this;
    }

    PSROIPoolingLayerDef& coords(const std::string& name, size_t index = 0) {
        coords_port_ = PortInfo(name, index);
        return *this;
    }

    TestNetwork& build();
};
