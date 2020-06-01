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

struct ProposalParams final {
    ProposalParams& feat_stride(size_t feat_stride) {
        feat_stride_ = feat_stride;
        return *this;
    }

    ProposalParams& base_size(size_t base_size) {
        base_size_ = base_size;
        return *this;
    }

    ProposalParams& min_size(size_t min_size) {
        min_size_ = min_size;
        return *this;
    }

    ProposalParams& pre_nms_topn(int pre_nms_topn) {
        pre_nms_topn_ = pre_nms_topn;
        return *this;
    }

    ProposalParams& post_nms_topn(int post_nms_topn) {
        post_nms_topn_ = post_nms_topn;
        return *this;
    }

    ProposalParams& nms_thresh(float nms_thresh) {
        nms_thresh_ = nms_thresh;
        return *this;
    }

    ProposalParams& framework(std::string framework) {
        framework_ = std::move(framework);
        return *this;
    }

    ProposalParams& scale(std::vector<float> scale) {
        scale_ = std::move(scale);
        return *this;
    }

    ProposalParams& ratio(std::vector<float> ratio) {
        ratio_ = std::move(ratio);
        return *this;
    }

    ProposalParams& normalize(bool normalize) {
        normalize_ = normalize;
        return *this;
    }

    ProposalParams& for_deformable(bool for_deformable) {
        for_deformable_ = for_deformable;
        return *this;
    }

    ProposalParams& clip_after_nms(bool clip_after_nms) {
        clip_after_nms_ = clip_after_nms;
        return *this;
    }

    ProposalParams& clip_before_nms(bool clip_before_nms) {
        clip_before_nms_ = clip_before_nms;
        return *this;
    }

    ProposalParams& box_scale(float box_size_scale) {
        box_size_scale_ = box_size_scale;
        return *this;
    }

    ProposalParams& box_coord_scale(float box_coordinate_scale) {
        box_coordinate_scale_ = box_coordinate_scale;
        return *this;
    }

    size_t feat_stride_ = 0;
    size_t base_size_   = 0;
    size_t min_size_    = 0;

    int pre_nms_topn_  = -1;
    int post_nms_topn_ = -1;

    float nms_thresh_           = 0.0;
    float box_coordinate_scale_ = 1.0;
    float box_size_scale_       = 1.0;

    bool normalize_       = false;
    bool clip_before_nms_ = true;
    bool clip_after_nms_  = false;
    bool for_deformable_  = false;

    std::string framework_;
    std::vector<float> scale_;
    std::vector<float> ratio_;
};

inline std::ostream& operator<<(std::ostream& os, const ProposalParams& p) {
    vpu::formatPrint(os, "[feat_stride:%v, base_size:%v, min_size:%v, pre_nms_topn:%v,"
                           "post_nms_topn:%v, nms_thresh:%v, box_coordinate_scale:%v,"
                           "box_size_scale:%v, normalize:%v, clip_before_nms:%v, clip_after_nms:%v,"
                           " for_deformable:%v, framework:%v, scale:%v, ratio:%v]",
                           p.feat_stride_, p.base_size_, p.min_size_, p.pre_nms_topn_,
                           p.post_nms_topn_, p.nms_thresh_, p.box_coordinate_scale_,
                           p.box_size_scale_, p.normalize_, p.clip_before_nms_, p.clip_after_nms_,
                           p.for_deformable_, p.framework_, p.scale_, p.ratio_);
    return os;
}

struct ProposalLayerDef final {
    TestNetwork&   net_;
    std::string    name_;
    ProposalParams params_;

    PortInfo cls_score_port_;
    PortInfo bbox_pred_port_;
    PortInfo img_info_port_;

    ProposalLayerDef(TestNetwork& net, std::string name, ProposalParams params)
        : net_(net), name_(std::move(name)), params_(std::move(params)) {
    }

    ProposalLayerDef& scores(const std::string& layer_name, size_t index = 0) {
        cls_score_port_ = PortInfo(layer_name, index);
        return *this;
    }

    ProposalLayerDef& boxDeltas(const std::string& layer_name, size_t index = 0) {
        bbox_pred_port_ = PortInfo(layer_name, index);
        return *this;
    }

    ProposalLayerDef& imgInfo(const std::string& layer_name, size_t index = 0) {
        img_info_port_ = PortInfo(layer_name, index);
        return *this;
    }

    TestNetwork& build();
};
