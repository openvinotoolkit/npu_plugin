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

#include "kmb_test_deconvolution_def.hpp"
#include "kmb_test_add_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/convolution.hpp>

#include "deconv_ref.hpp"

namespace {

BlobVector refDeconv(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto deconvLayer = std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolutionBackpropData>(layer);
    IE_ASSERT(deconvLayer != nullptr);

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

    unsigned groups = layer->input(1).get_shape()[0];

    // Only DW case is supported
    IE_ASSERT((groups == layer->input(0).get_shape()[1] && groups == layer->output(0).get_shape()[1]));

    // per channel shapes
    ngraph::Shape newInShape =  {1, 1, layer->input(0).get_shape().end()[-2], layer->input(0).get_shape().end()[-1]};
    ngraph::Shape newWShape =   {1, 1, layer->input(1).get_shape().end()[-2], layer->input(1).get_shape().end()[-1]};
    ngraph::Shape newOutShape = {1, 1, layer->output(0).get_shape().end()[-2], layer->output(0).get_shape().end()[-1]};

    for (unsigned i = 0; i < groups; ++i) {
        // offset for spatial dimensions, requires NCHW shape
        auto inOffset  = i * layer->input(0).get_shape().end()[-2]  * layer->input(0).get_shape().end()[-1];
        auto wOffset   = i * layer->input(1).get_shape().end()[-2]  * layer->input(1).get_shape().end()[-1];
        auto outOffset = i * layer->output(0).get_shape().end()[-2] * layer->output(0).get_shape().end()[-1];

        // Use convolution with proper strides, input dilation and padding, to be tested with different parameters
        ngraph::runtime::reference::convolution(
            inputPtr   + inOffset,
            weightsPtr + wOffset,
            outputPtr  + outOffset,
            newInShape, newWShape, newOutShape,
            {1, 1}, // kernel stride now is just ones, we use other paramaters to emulate deconv
            deconvLayer->get_dilations(), // use normal dilation
            // pads begin[spatial dim] += k stiride[spatial dim]/2
            {unsigned(deconvLayer->get_pads_begin()[0] + deconvLayer->get_strides()[0] / 2),
                unsigned(deconvLayer->get_pads_begin()[1] + deconvLayer->get_strides()[1] / 2)},
            {unsigned(deconvLayer->get_pads_end()[0] + deconvLayer->get_strides()[0] / 2),
                unsigned(deconvLayer->get_pads_end()[1] + deconvLayer->get_strides()[1] / 2)},
            // input dilation = 1 + k stride / 2
            ngraph::Strides {unsigned(1 + deconvLayer->get_strides()[0] / 2),
                unsigned(1 + deconvLayer->get_strides()[1] / 2)});
    }

    return {output};
};

}  // namespace

TestNetwork& DeconvolutionLayerDef::build() {

    const auto strides = ngraph::Strides {params._strides.y, params._strides.x};
    const auto dilations = ngraph::Strides {params._dilation.y, params._dilation.x};
    const auto padsBegin = ngraph::CoordinateDiff {params._pad.top, params._pad.left};
    const auto padsEnd = ngraph::CoordinateDiff {params._pad.bottom, params._pad.right};
    const auto out_padding = ngraph::CoordinateDiff {0, 0};

    const auto deconvNode =
        std::make_shared<ngraph::op::v1::GroupConvolutionBackpropData>(
            testNet.getPort(inputPort),
            testNet.getPort(weightsPort),
            strides,
            padsBegin,
            padsEnd,
            dilations,
            ngraph::op::PadType::EXPLICIT,
            out_padding);

    return testNet.addLayer(name, deconvNode, refDeconv);
}

TensorDesc getDeconvDwWeightsDesc(const DeconvolutionParams& params, Precision precision) {
    return {precision, {params._group , 1, 1, params._kernel.y, params._kernel.x}, Layout::GOIHW};
}

