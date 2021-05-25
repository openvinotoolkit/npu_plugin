//
// Copyright 2019 Intel Corporation.
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

#include "kmb_test_strided_slice_def.hpp"

#include <blob_factory.hpp>
#include <ngraph/runtime/reference/strided_slice.hpp>


namespace {

ngraph::SlicePlan get_slice_plan(std::shared_ptr<ngraph::op::v1::StridedSlice> slice) {
    auto convert_mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        ngraph::AxisSet axis_set{};
        for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
            if (mask[i] == 1)
                axis_set.emplace(i);
        }
        return axis_set;
    };

    auto data = slice->input_value(0).get_node_shared_ptr();
    auto begin = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
    auto end = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
    auto strides = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
    if (!begin || !end || !strides || slice->input(0).get_partial_shape().is_dynamic())
        return ngraph::SlicePlan();

    auto begin_vec = begin->cast_vector<int64_t>();
    auto end_vec = end->cast_vector<int64_t>();
    auto strides_vec = strides->cast_vector<int64_t>();
    const auto begin_mask = convert_mask_to_axis_set(slice->get_begin_mask());
    const auto end_mask = convert_mask_to_axis_set(slice->get_end_mask());

    ngraph::SlicePlan plan = ngraph::make_slice_plan(slice->input(0).get_shape(),
                                                     begin_vec,
                                                     end_vec,
                                                     strides_vec,
                                                     begin_mask,
                                                     end_mask,
                                                     convert_mask_to_axis_set(slice->get_new_axis_mask()),
                                                     convert_mask_to_axis_set(slice->get_shrink_axis_mask()),
                                                     convert_mask_to_axis_set(slice->get_ellipsis_mask()));
    return plan;
}

BlobVector refStridedSlice(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 4);

    const auto stridedSliceLayer = std::dynamic_pointer_cast<ngraph::op::v1::StridedSlice>(layer);
    IE_ASSERT(stridedSliceLayer != nullptr);

    const auto input = inputs.at(0);
    const auto output = vpux::makeSplatBlob(input->getTensorDesc(), 0.0f);

    const auto inputPtr = input->cbuffer().as<const float*>();
    auto outputPtr = output->buffer().as<float*>();

    IE_ASSERT(inputPtr != nullptr);
    IE_ASSERT(outputPtr != nullptr);

    const auto slicePlan = get_slice_plan(stridedSliceLayer);

    ngraph::runtime::reference::strided_slice(
            reinterpret_cast<const char*>(inputPtr),
            reinterpret_cast<char*>(outputPtr),
            layer->input(0).get_shape(), slicePlan, input->element_size());

    return {output};
};

} // namespace

TestNetwork& StridedSliceLayerDef::build() {
     const auto node =
        std::make_shared<ngraph::op::v1::StridedSlice>(
                    testNet.getPort(inputPort),
                    testNet.getPort(beginsPort),
                    testNet.getPort(endsPort),
                    testNet.getPort(stridesPort),
                    params.beginMask,
                    params.endMask,
                    params.newAxisMask,
                    params.shrinkAxisMask,
                    params.ellipsisAxisMask);

    return testNet.addLayer(name, node, refStridedSlice);
}
