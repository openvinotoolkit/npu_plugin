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

#include "kmb_test_convolution_def.hpp"
#include "kmb_test_add_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/convolution.hpp>

namespace {

BlobVector refConv(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto convLayer = std::dynamic_pointer_cast<ngraph::op::v1::Convolution>(layer);
    IE_ASSERT(convLayer != nullptr);

    const auto input = inputs.at(0);
    const auto weights = inputs.at(1);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    const auto inputPtr = input->cbuffer().as<const float*>();
    const auto weightsPtr = weights->cbuffer().as<const float*>();
    auto outputPtr = output->buffer().as<float*>();

    IE_ASSERT(inputPtr != nullptr);
    IE_ASSERT(weightsPtr != nullptr);
    IE_ASSERT(outputPtr != nullptr);

    ngraph::runtime::reference::convolution(
        inputPtr, weightsPtr, outputPtr,
        layer->input(0).get_shape(), layer->input(1).get_shape(), layer->output(0).get_shape(),
        convLayer->get_strides(), convLayer->get_dilations(), convLayer->get_pads_begin(), convLayer->get_pads_end());

    return {output};
};

}  // namespace

TestNetwork& ConvolutionLayerDef::build() {
    const auto strides = ngraph::Strides {params._strides.y, params._strides.x};
    const auto dilation = ngraph::Strides {params._dilation.y, params._dilation.x};
    const auto padsBegin = ngraph::CoordinateDiff {params._pad.top, params._pad.left};
    const auto padsEnd = ngraph::CoordinateDiff {params._pad.bottom, params._pad.right};

    const auto convNode =
        std::make_shared<ngraph::op::v1::Convolution>(
            testNet.getPort(inputPort), testNet.getPort(weightsPort),
            strides, padsBegin, padsEnd, dilation);

    if (biasesPort.layerName.empty()) {
        return testNet.addLayer(name, convNode, refConv);
    } else {
        testNet.addLayer(name + "_conv", convNode, refConv);

        return
            testNet.addLayer<AddLayerDef>(name)
                .input1(name + "_conv")
                .input2(biasesPort.layerName, biasesPort.index)
                .build();
    }
}

TensorDesc getConvWeightsDesc(const ConvolutionParams& params, size_t inChannels, Precision precision) {
    return {precision, {params._outChannels, inChannels, params._kernel.y, params._kernel.x}, Layout::NCHW};
}

TensorDesc getConvBiasesDesc(const ConvolutionParams& params, Precision precision) {
    return {precision, {1, params._outChannels, 1, 1}, Layout::NCHW};
}
