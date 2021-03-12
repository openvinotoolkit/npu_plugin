//
// Copyright 2021 Intel Corporation.
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

#include "ngraph_mcm_frontend/passes/yolo_v4_hacks.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/pattern/op/true.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "ngraph/rt_info.hpp"

static bool process_interp(const std::shared_ptr<ngraph::op::v0::Interpolate>& interp) {
    const auto input_values = interp->input_values();
    const auto strides = ngraph::Strides{1, 1};
    const auto pads_begin = ngraph::Shape{0, 0};
    const auto pads_end = ngraph::Shape{0, 0};
    const auto kernel = ngraph::Shape{1, 1};
    const auto rounding_mode = ngraph::op::RoundingType::FLOOR;
    const auto auto_pad = ngraph::op::PadType::EXPLICIT;
    const auto pre_max_pool_1 =
            std::make_shared<ngraph::op::v1::MaxPool>(input_values.at(0).get_node_shared_ptr()->output(0), strides,
                                                      pads_begin, pads_end, kernel, rounding_mode, auto_pad);

    const auto pre_max_pool_2 = std::make_shared<ngraph::op::v1::MaxPool>(
            pre_max_pool_1->output(0), strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);

    ngraph::OutputVector new_inputs = {};
    new_inputs.push_back(pre_max_pool_2->output(0));
    for (size_t i = 1; i < input_values.size(); i++) {
        new_inputs.push_back(input_values.at(i));
    }

    const auto new_interp = interp->clone_with_new_inputs(new_inputs);
    const auto post_max_pool_1 = std::make_shared<ngraph::op::v1::MaxPool>(new_interp->output(0), strides, pads_begin,
                                                                           pads_end, kernel, rounding_mode, auto_pad);

    const auto post_max_pool_2 = std::make_shared<ngraph::op::v1::MaxPool>(
            post_max_pool_1->output(0), strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);

    ngraph::replace_node(interp, post_max_pool_2);
    return true;
}

static bool process_concat(const std::shared_ptr<ngraph::op::Concat>& concat) {
    const auto input_values = concat->input_values();
    const auto strides = ngraph::Strides{1, 1};
    const auto pads_begin = ngraph::Shape{0, 0};
    const auto pads_end = ngraph::Shape{0, 0};
    const auto kernel = ngraph::Shape{1, 1};
    const auto rounding_mode = ngraph::op::RoundingType::FLOOR;
    const auto auto_pad = ngraph::op::PadType::EXPLICIT;

    std::vector<std::shared_ptr<ngraph::op::v1::MaxPool>> max_pools = {};
    for (size_t i = 0; i < input_values.size(); i++) {
        max_pools.push_back(std::make_shared<ngraph::op::v1::MaxPool>(input_values.at(i), strides, pads_begin, pads_end,
                                                                      kernel, rounding_mode, auto_pad));
    }

    ngraph::OutputVector new_inputs = {};
    for (size_t i = 0; i < input_values.size(); i++) {
        new_inputs.push_back(max_pools.at(i)->output(0));
    }

    const auto new_concat = concat->clone_with_new_inputs(new_inputs);

    ngraph::replace_node(concat, new_concat);
    return true;
}

/// FIXME use inference engine subroutine
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16 0x7C00U

using ie_fp16 = int16_t;

inline float asfloat(uint32_t v) {
    // Both type-punning casts and unions are UB per C++ spec
    // But compilers usually only break code with casts
    union {
        float f;
        uint32_t i;
    };
    i = v;
    return f;
}

ie_fp16 f32tof16(float x) {
    // create minimal positive normal f16 value in f32 format
    // exp:-14,mantissa:0 -> 2^-14 * 1.0
    static float min16 = asfloat((127 - 14) << 23);

    // create maximal positive normal f16 value in f32 and f16 formats
    // exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermediate and output result
    // the union is used to simplify representation changing
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get sign in 16bit format
    uint32_t s = (v.u >> 16) & 0x8000;  // sign 16:  00000000 00000000 10000000 00000000

    // make it abs
    v.u &= 0x7FFFFFFF;  // abs mask: 01111111 11111111 11111111 11111111

    // check NAN and INF
    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return s | (v.u >> (23 - 10)) | 0x0200;  // return NAN f16
        } else {
            return s | (v.u >> (23 - 10));  // return INF f16
        }
    }

    // to make f32 round to nearest f16
    // create halfULP for f16 and add it to origin value
    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    // if input value is not fit normalized f16 then return 0
    // denormals are not covered by this code and just converted to 0
    if (v.f < min16 * 0.5F) {
        return s;
    }

    // if input value between min16/2 and min16 then return min16
    if (v.f < min16) {
        return s | (1 << 10);
    }

    // if input value more than maximal allowed value for f16
    // then return this maximal value
    if (v.f >= max16) {
        return max16f16 | s;
    }

    // change exp bias from 127 to 15
    v.u -= ((127 - 15) << 23);

    // round to f16
    v.u >>= (23 - 10);

    return v.u | s;
}

bool process_fq(const std::shared_ptr<ngraph::op::FakeQuantize>& fq_node) {
    const auto input_values = fq_node->input_values();
    const auto first_input = input_values.at(0).get_node_shared_ptr();
    const auto mish = std::dynamic_pointer_cast<ngraph::op::v4::Mish>(first_input);
    if (!mish) {
        return false;
    }
    auto constant = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_values.at(1).get_node_shared_ptr());
    if (!constant) {
        return false;
    }

    const auto fp32_vec = constant->cast_vector<float>();
    if (fp32_vec.size() != 1 && fp32_vec.at(0) >= -5.5f) {
        return false;
    }

    std::vector<ie_fp16> lo_weights = {f32tof16(-5.5f)};

    const auto in_lo =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, ngraph::Shape{}, lo_weights.data());

    const auto out_lo =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, ngraph::Shape{}, lo_weights.data());

    ngraph::OutputVector new_inputs = {input_values.at(0), in_lo, input_values.at(2), out_lo, input_values.at(4)};

    const auto new_fq = fq_node->clone_with_new_inputs(new_inputs);
    ngraph::replace_node(fq_node, new_fq);

    return true;
}

ngraph::pass::YoloV4Hacks::YoloV4Hacks() {
    ngraph::graph_rewrite_callback replace_cb = [](pattern::Matcher& m) {
        auto match_root = m.get_match_root();
        const auto interp = std::dynamic_pointer_cast<ngraph::op::v0::Interpolate>(match_root);
        if (interp) {
            return process_interp(interp);
        }

        const auto concat = std::dynamic_pointer_cast<ngraph::op::Concat>(match_root);
        if (concat) {
            return process_concat(concat);
        }

        const auto fq = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(match_root);
        if (fq) {
            return process_fq(fq);
        }

        return false;
    };

    auto all_nodes = std::make_shared<ngraph::pattern::op::True>();
    auto match = std::make_shared<ngraph::pattern::Matcher>(all_nodes, "mcmAdaptation.YoloV4Hacks");
    this->register_matcher(match, replace_cb);
}
