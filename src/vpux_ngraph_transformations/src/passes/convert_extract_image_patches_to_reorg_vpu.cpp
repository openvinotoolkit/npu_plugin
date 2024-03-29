//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/convert_extract_image_patches_to_reorg_vpu.hpp"
#include <memory>

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace vpux {
namespace passes {

ConvertExtractImagePatchesToReorgYoloVPU::ConvertExtractImagePatchesToReorgYoloVPU() {
    auto image = std::make_shared<ov::pass::pattern::op::Label>(ov::element::f32, ov::Shape{1, 1, 1, 1});
    auto eip = std::make_shared<ov::opset3::ExtractImagePatches>(image, ov::Shape{1, 1}, ov::Strides{1, 1},
                                                                 ov::Shape{1, 1}, ov::op::PadType::VALID);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto extract_image_patches = std::dynamic_pointer_cast<ov::opset3::ExtractImagePatches>(m.get_match_root());

        /*
         * In this transformation we replace ExtractImagePatches operation to ReorgYolo operation
         * if ExtractImagePatches operation attributes obey the following conditions:
         *
         * EIP.sizes = EIP.strides
         * EIP.rates = {1, 1}
         * EIP.PadType = VALID
         * Spatial dimensions of input tensor must be divisible by EIP.strides
         *
         */

        if (!extract_image_patches || transformation_callback(extract_image_patches)) {
            return false;
        }

        if (extract_image_patches->get_strides() != extract_image_patches->get_sizes()) {
            return false;
        }

        auto p_shape_input = extract_image_patches->get_input_partial_shape(0);
        auto sizes = extract_image_patches->get_sizes();
        auto strides = extract_image_patches->get_strides();
        auto rates = extract_image_patches->get_rates();

        // Check that ExtractImagePatches input have static shape and rank == 4
        if (!p_shape_input.rank().is_static() || p_shape_input.rank().get_length() != 4) {
            return false;
        }

        // Check that ExtractImagePatches input spatial dimensions are not dynamic
        if (p_shape_input[2].is_dynamic() || p_shape_input[3].is_dynamic()) {
            return false;
        }

        // Check that ExtractImagePatches input spatial dimensions are divisible by EIP.strides
        if (p_shape_input[2].get_length() % strides[0] != 0 || p_shape_input[3].get_length() % strides[1] != 0) {
            return false;
        }

        // Check that EIP.sizes = EIP.strides
        if (sizes[0] != strides[0] || sizes[1] != strides[1]) {
            return false;
        }

        // Check that EIP.rates = {1, 1}
        if (rates[0] != 1 || rates[1] != 1) {
            return false;
        }

        auto reorg_yolo = std::make_shared<ov::opset3::ReorgYolo>(extract_image_patches->input(0).get_source_output(),
                                                                  ov::Strides{extract_image_patches->get_strides()});

        reorg_yolo->set_friendly_name(extract_image_patches->get_friendly_name());
        ov::copy_runtime_info(extract_image_patches, reorg_yolo);
        ov::replace_node(extract_image_patches, reorg_yolo);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(eip, "ConvertExtractImagePatchesToReorgYolo");
    register_matcher(m, callback);
}

}  // namespace passes
}  // namespace vpux
