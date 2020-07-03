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

#include "kmb_test_reshape_def.hpp"

#include <blob_factory.hpp>

namespace {
    BlobVector refReshape(const TestNetwork::NodePtr& layer,
                          const BlobVector& inputs,
                          const TestNetwork&) {
        if (inputs.size() != 2) {
            THROW_IE_EXCEPTION << "Incorrect number of inputs";
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
