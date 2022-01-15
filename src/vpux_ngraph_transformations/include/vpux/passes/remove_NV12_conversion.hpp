//
// Copyright Intel Corporation.
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

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>
#include <openvino/op/nv12_to_bgr.hpp>
#include "ngraph/op/nv12_to_rgb.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux_compiler.hpp"

namespace vpux {
namespace pass {

InferenceEngine::Precision importElemType(const ngraph::element::Type& elemType) {
    if (elemType == ngraph::element::f64) {
        return InferenceEngine::Precision::FP64;
    } else if (elemType == ngraph::element::f32) {
        return InferenceEngine::Precision::FP32;
    } else if (elemType == ngraph::element::f16) {
        return InferenceEngine::Precision::FP16;
    } else if (elemType == ngraph::element::bf16) {
        return InferenceEngine::Precision::BF16;
    } else if (elemType == ngraph::element::i64) {
        return InferenceEngine::Precision::I64;
    } else if (elemType == ngraph::element::u64) {
        return InferenceEngine::Precision::U64;
    } else if (elemType == ngraph::element::i32) {
        return InferenceEngine::Precision::I32;
    } else if (elemType == ngraph::element::u32) {
        return InferenceEngine::Precision::U32;
    } else if (elemType == ngraph::element::i16) {
        return InferenceEngine::Precision::I16;
    } else if (elemType == ngraph::element::u16) {
        return InferenceEngine::Precision::U16;
    } else if (elemType == ngraph::element::i8) {
        return InferenceEngine::Precision::I8;
    } else if (elemType == ngraph::element::u8) {
        return InferenceEngine::Precision::U8;
    } else if (elemType == ngraph::element::boolean) {
        return InferenceEngine::Precision::BOOL;
    } else {
        return InferenceEngine::Precision::UNSPECIFIED;
    }
}

template <typename T>
class RemoveNV12Conversion final : public ngraph::pass::FunctionPass {
public:
    explicit RemoveNV12Conversion(std::vector<vpux::VPUXPreProcessInfo::Ptr>& preProcInfo): _preProcInfo(preProcInfo){};

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

                _preProcInfo.push_back(std::make_shared<VPUXPreProcessInfo>(
                        param->get_friendly_name(), InferenceEngine::ColorFormat::NV12, getColorSpace(nv12),
                        InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR,
                        InferenceEngine::TensorDesc(importElemType(nv12->input(0).get_element_type()),
                                                    nv12->input(0).get_partial_shape().get_shape(),
                                                    InferenceEngine::Layout::NHWC)));
                return true;
            }
        }
        return false;
    };

private:
    static InferenceEngine::ColorFormat getColorSpace(const std::shared_ptr<ov::op::v8::NV12toRGB>&) {
        return InferenceEngine::ColorFormat::RGB;
    }
    static InferenceEngine::ColorFormat getColorSpace(const std::shared_ptr<ov::op::v8::NV12toBGR>&) {
        return InferenceEngine::ColorFormat::BGR;
    }

    std::vector<vpux::VPUXPreProcessInfo::Ptr>& _preProcInfo;
};

}  // namespace pass
}  // namespace vpux
