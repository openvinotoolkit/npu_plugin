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

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct ROIPoolingParams final {
    size_t pooled_w_     = 1u;
    size_t pooled_h_     = 1u;
    float spatial_scale_ = 0.0f;
    std::string mode_;

    ROIPoolingParams& pooled_w(size_t pooled_w) {
         pooled_w_ =  pooled_w;
        return *this;
    }

    ROIPoolingParams& pooled_h(size_t pooled_h) {
        pooled_h_ = pooled_h;
        return *this;
    }

    ROIPoolingParams& spatial_scale(float spatial_scale) {
        spatial_scale_ = spatial_scale;
        return *this;
    }

    ROIPoolingParams& mode(std::string mode) {
        mode_ = std::move(mode);
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const ROIPoolingParams& p) {
        vpu::formatPrint(os, "[pooled_w:%v, pooled_h:%v, spatial_scale:%v, mode:%v]",
                p.pooled_w_, p.pooled_h_, p.spatial_scale_, p.mode_);
	return os;
}

struct ROIPoolingLayerDef final {
    TestNetwork& net_;
    std::string name_;
    ROIPoolingParams params_;

    PortInfo input_port_;
    PortInfo coords_port_;

    ROIPoolingLayerDef(TestNetwork& net, std::string name, ROIPoolingParams params)
        : net_(net), name_(std::move(name)), params_(std::move(params)) {
    }

    ROIPoolingLayerDef& input(const std::string& name, size_t index = 0) {
        input_port_ = PortInfo(name, index);
        return *this;
    }

    ROIPoolingLayerDef& coords(const std::string& name, size_t index = 0) {
        coords_port_ = PortInfo(name, index);
        return *this;
    }

    TestNetwork& build();
};
