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

#include "kmb_test_roipooling_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/op/roi_pooling.hpp>

namespace {

BlobVector refROIPooling(const TestNetwork::NodePtr& layer, const BlobVector& /*inputs*/, const TestNetwork&) {
    auto shape = layer->get_shape();
    IE_ASSERT(shape.size() == 4);
    SizeVector out_dims{shape[0], shape[1], shape[2], shape[3]};
    auto out_desc = TensorDesc(Precision::FP32, out_dims, Layout::NCHW);
    auto output = make_blob_with_precision(out_desc);
    output->allocate();
    float* dst_data = output->buffer().as<float*>();

    for (auto i = 0u; i < shape[0] * shape[1] * shape[2] * shape[3]; ++i)
        dst_data[i] = 0.0f;

    return {output};

};

}  // namespace

TestNetwork& ROIPoolingLayerDef::build() {
    auto input  = net_.getPort(input_port_);
    auto coords = net_.getPort(coords_port_);
    const ngraph::Shape output_size { params_.pooled_h_, params_.pooled_w_ };

    auto roi  = std::make_shared<ngraph::op::ROIPooling>(input,
                                                         coords,
                                                         output_size,
                                                         params_.spatial_scale_,
                                                         params_.mode_);
    return net_.addLayer(name_, roi, refROIPooling);
}
