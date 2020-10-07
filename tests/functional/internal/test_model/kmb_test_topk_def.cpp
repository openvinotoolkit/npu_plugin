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

#include "kmb_test_topk_def.hpp"

#include <blob_factory.hpp>
#include <ngraph/runtime/reference/topk.hpp>

namespace {

BlobVector refTopK(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto topkLayer = std::dynamic_pointer_cast<ngraph::op::v1::TopK>(layer);
    IE_ASSERT(topkLayer != nullptr);

    const auto input = inputs.at(0);
    const auto scalar = inputs.at(1);
    const auto axis = topkLayer->get_axis();
    const auto k = topkLayer->get_k();
    const auto sortType = topkLayer->get_sort_type();
    bool computeMax = (topkLayer->get_mode() == ngraph::op::v1::TopK::Mode::MAX);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));

    const auto outputIndices = make_blob_with_precision(outDesc);
    outputIndices->allocate();
    const auto outputValues = make_blob_with_precision(outDesc);
    outputValues->allocate();

    const auto inputPtr = input->cbuffer().as<const float*>();
    const auto scalarPtr = scalar->cbuffer().as<const float*>();

    auto outputIndicesPtr = outputIndices->buffer().as<float*>();
    auto outputValuesPtr = outputValues->buffer().as<float*>();

    IE_ASSERT(inputPtr != nullptr);
    IE_ASSERT(scalarPtr != nullptr);
    IE_ASSERT(outputIndicesPtr != nullptr);
    IE_ASSERT(outputValuesPtr != nullptr);

    std::fill_n(outputIndicesPtr, outputIndices->size(), static_cast<float>(0.0));
    std::fill_n(outputValuesPtr, outputValues->size(), static_cast<float>(0.0));

    ngraph::runtime::reference::topk(inputPtr, outputIndicesPtr, outputValuesPtr, layer->input(0).get_shape(),
        layer->output(0).get_shape(), axis, k, computeMax, sortType);

    return {outputValues, outputIndices};
};

}  // namespace

TestNetwork& TopKLayerDef::build() {
    const auto node = std::make_shared<ngraph::op::v1::TopK>(
        testNet.getPort(inputPort), testNet.getPort(scalarKPort), params._axis, params._modeType, params._sortType);

    return testNet.addLayer(name, node, refTopK);
}

std::ostream& operator<<(std::ostream& os, const TopKParams& p) {
    vpu::formatPrint(os, "[_axis:%v,_mode:%v,_sort:%v]", p._axis, p._modeType, p._sortType);
    return os;
}
