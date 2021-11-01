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

// clang-format on

#include "ngraph_mcm_frontend/passes/segnet_workaround.hpp"
#include <fstream>
#include <ngraph/op/reorg_yolo.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

// it's a workaround for segnet.
SegnetWorkaround::SegnetWorkaround() {
    auto input = ngraph::pattern::any_input();
    auto VariadicSplit1 =
            std::make_shared<ngraph::op::v1::VariadicSplit>(input, ngraph::pattern::wrap_type<ngraph::op::Constant>(),
                                                            ngraph::pattern::wrap_type<ngraph::op::Constant>());
    std::cout << "VariadicSplit1" << std::endl;
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(
            VariadicSplit1->output(0), ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(
            reshape1, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(
            reshape2, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto ScatterNDUpdate1 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ngraph::pattern::wrap_type<ngraph::op::Constant>(),
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})), reshape3);

    // auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(
    //         VariadicSplit1->output(0), ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    // auto reshape5 = std::make_shared<ngraph::opset1::Reshape>(
    //         reshape4, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    // auto reshape6 = std::make_shared<ngraph::opset1::Reshape>(
    //         reshape5, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto ScatterNDUpdate2 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ScatterNDUpdate1,
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})),
            ngraph::pattern::any_input());

    auto ScatterNDUpdate3 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ScatterNDUpdate2,
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})),
            ngraph::pattern::any_input());

    auto ScatterNDUpdate4 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ScatterNDUpdate3,
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})),
            ngraph::pattern::any_input());

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto reAllocate_input = pattern_to_output.at(input);

        auto first_ScatterNDUpdate = std::dynamic_pointer_cast<ngraph::op::v3::ScatterNDUpdate>(
                pattern_to_output.at(ScatterNDUpdate1).get_node_shared_ptr());
        auto idx = std::dynamic_pointer_cast<ngraph::op::Constant>(
                first_ScatterNDUpdate->input_value(1).get_node_shared_ptr());
        auto scaleshift_scale_data = idx->cast_vector<uint8_t>();
        auto shape = first_ScatterNDUpdate->get_input_shape(1);

        std::cout << "shape: " << std::endl;
        for (unsigned i = 0; i < shape.size(); i++) {
            std::cout << shape[i] << std::endl;
        }

        // std::cout << "value " << std::endl;
        // for (unsigned i = 0; i < scaleshift_scale_data.size() / 4; i++) {
        //     std::cout << scaleshift_scale_data[i * 4] << " " << scaleshift_scale_data[i * 4 + 1] << " "
        //               << scaleshift_scale_data[i * 4 + 2] << " " << scaleshift_scale_data[i * 4 + 3] << std::endl;
        // }

        std::ofstream fout1("ScatterNDUpdate1.dat", std::ios::out | std::ios::binary);
        fout1.write((char*)&scaleshift_scale_data[0], scaleshift_scale_data.size() * sizeof(scaleshift_scale_data[0]));
        fout1.close();

        auto second_ScatterNDUpdate = std::dynamic_pointer_cast<ngraph::op::v3::ScatterNDUpdate>(
                pattern_to_output.at(ScatterNDUpdate2).get_node_shared_ptr());
        auto idx2 = std::dynamic_pointer_cast<ngraph::op::Constant>(
                second_ScatterNDUpdate->input_value(1).get_node_shared_ptr());
        auto scaleshift_scale_data2 = idx2->cast_vector<uint8_t>();

        std::ofstream fout2("ScatterNDUpdate2.dat", std::ios::out | std::ios::binary);
        fout2.write((char*)&scaleshift_scale_data2[0],
                    scaleshift_scale_data2.size() * sizeof(scaleshift_scale_data2[0]));
        fout2.close();

        auto third_ScatterNDUpdate = std::dynamic_pointer_cast<ngraph::op::v3::ScatterNDUpdate>(
                pattern_to_output.at(ScatterNDUpdate3).get_node_shared_ptr());
        auto idx3 = std::dynamic_pointer_cast<ngraph::op::Constant>(
                third_ScatterNDUpdate->input_value(1).get_node_shared_ptr());
        auto scaleshift_scale_data3 = idx3->cast_vector<uint8_t>();

        std::ofstream fout3("ScatterNDUpdate3.dat", std::ios::out | std::ios::binary);
        fout3.write((char*)&scaleshift_scale_data3[0],
                    scaleshift_scale_data3.size() * sizeof(scaleshift_scale_data3[0]));
        fout3.close();

        auto fourth_ScatterNDUpdate = std::dynamic_pointer_cast<ngraph::op::v3::ScatterNDUpdate>(
                pattern_to_output.at(ScatterNDUpdate4).get_node_shared_ptr());
        auto idx4 = std::dynamic_pointer_cast<ngraph::op::Constant>(
                fourth_ScatterNDUpdate->input_value(1).get_node_shared_ptr());
        auto scaleshift_scale_data4 = idx4->cast_vector<uint8_t>();

        std::ofstream fout4("ScatterNDUpdate4.dat", std::ios::out | std::ios::binary);
        fout4.write((char*)&scaleshift_scale_data4[0],
                    scaleshift_scale_data4.size() * sizeof(scaleshift_scale_data4[0]));
        fout4.close();

        // if (first_transpose == nullptr) {
        //     return false;
        // }

        // auto last_transpose = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
        //         pattern_to_output.at(transpose5).get_node_shared_ptr());
        // if (last_transpose == nullptr) {
        //     return false;
        // }
        std::cout << "find pattern" << std::endl;

        auto reAllocate = std::make_shared<ngraph::op::v0::DepthToSpace>(reAllocate_input, "DEPTH_FIRST", 2);

        // ngraph::copy_runtime_info(
        //         {
        //                 pattern_to_output.at(transpose1).get_node_shared_ptr(),
        //                 pattern_to_output.at(reshape1).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose2).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose3).get_node_shared_ptr(),
        //                 pattern_to_output.at(reshape2).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose4).get_node_shared_ptr(),
        //                 pattern_to_output.at(reshape3).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose5).get_node_shared_ptr(),
        //         },
        //         depthToSpace);
        ngraph::replace_node(m.get_match_root(), reAllocate);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ScatterNDUpdate4, "SegnetWorkaround");
    register_matcher(m, matcher_pass_callback);
}
