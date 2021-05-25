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
