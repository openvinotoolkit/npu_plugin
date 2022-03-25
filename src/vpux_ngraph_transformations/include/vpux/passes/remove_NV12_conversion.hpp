//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>
#include <openvino/op/nv12_to_bgr.hpp>
#include "ngraph/op/nv12_to_rgb.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux_compiler.hpp"

namespace vpux {
namespace pass {

template <typename T>
class RemoveNV12Conversion final : public ngraph::pass::FunctionPass {
public:
    explicit RemoveNV12Conversion(std::vector<vpux::PreProcessInfo>& preProcInfo): _preProcInfo(preProcInfo){};

    bool run_on_function(std::shared_ptr<ngraph::Function> f) final {
        for (const auto& node : f->get_ordered_ops()) {
            if (auto nv12 = std::dynamic_pointer_cast<T>(node)) {
                auto inputs = nv12->input_values();
                if (inputs.size() > 1)
                    continue;
                if (!std::all_of(inputs.begin(), inputs.end(), [](ov::Output<ov::Node> output) {
                        return std::dynamic_pointer_cast<ov::op::v0::Parameter>(output.get_node_shared_ptr()) &&
                               output.get_target_inputs().size() == 1;
                    })) {
                    continue;
                }
                auto param = std::make_shared<ov::op::v0::Parameter>(nv12->output(0).get_element_type(),
                                                                     nv12->output(0).get_partial_shape());
                param->set_friendly_name(nv12->get_input_node_shared_ptr(0)->get_friendly_name());
                ov::replace_node(nv12, param);
                auto old_param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(inputs.at(0).get_node_shared_ptr());
                f->replace_parameter(f->get_parameter_index(old_param), {param});

                _preProcInfo.push_back(vpux::PreProcessInfo(param->get_friendly_name(), PreProcessColorSpace::NV12,
                                                            getColorSpace(nv12),
                                                            PreProcessResizeAlgorithm::RESIZE_BILINEAR));
                return true;
            }
        }
        return false;
    };

private:
    static PreProcessColorSpace getColorSpace(const std::shared_ptr<ov::op::v8::NV12toRGB>&) {
        return vpux::PreProcessColorSpace::RGB;
    }
    static PreProcessColorSpace getColorSpace(const std::shared_ptr<ov::op::v8::NV12toBGR>&) {
        return vpux::PreProcessColorSpace::BGR;
    }

    std::vector<vpux::PreProcessInfo> _preProcInfo;
};

}  // namespace pass
}  // namespace vpux
