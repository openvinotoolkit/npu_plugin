//
// Copyright 2020 Intel Corporation.
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

#include "kmb_test_reshape_def.hpp"

#include <blob_factory.hpp>

namespace {
    BlobVector refReshape(const TestNetwork::NodePtr& layer,
                          const BlobVector& inputs,
                          const TestNetwork&) {
        if (inputs.size() != 2) {
            IE_THROW() << "Incorrect number of inputs";
        }

        const auto& in_desc = inputs[0]->getTensorDesc();
        const auto& out_dims = layer->output(0).get_shape();

        auto output = make_blob_with_precision(TensorDesc(in_desc.getPrecision(), out_dims,
                                                          TensorDesc::getLayoutByDims(out_dims)));
        output->allocate();
        IE_ASSERT(in_desc.getPrecision() == Precision::FP32);
        std::copy_n(inputs[0]->buffer().as<float*>(), inputs[0]->size(), output->buffer().as<float*>());

        return {output};
    }

}  // namespace

TestNetwork& ReshapeLayerDef::build() {
    auto in    = net_.getPort(in_port_);
    auto shape = net_.getPort(shape_port_);

    auto reshape = std::make_shared<ngraph::op::v1::Reshape>(in, shape, true);
    return net_.addLayer(name_, reshape, refReshape);
}
