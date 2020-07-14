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

#include "kmb_test_permute_def.hpp"

#include <blob_factory.hpp>

namespace {

void transpose(const Blob::Ptr& input, const int64_t* order, Blob::Ptr& output) {
    const auto& dims = input->getTensorDesc().getDims();
    const size_t kNumDims = dims.size();
    IE_ASSERT(kNumDims == 4);

    SizeVector in_strides;
    in_strides.reserve(kNumDims);
    size_t total_stride = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

    for (size_t i = 0; i < kNumDims; ++i) {
        total_stride /= dims[i];
        in_strides.push_back(total_stride);
    }

    SizeVector out_dims;
    out_dims.reserve(kNumDims);
    for (size_t i = 0; i < kNumDims; ++i) {
        out_dims.push_back(dims[order[i]]);
    }

    IE_ASSERT(input->getTensorDesc().getPrecision() == Precision::FP32 &&
              output->getTensorDesc().getPrecision() == Precision::FP32);

    float* src = input->buffer().as<float*>();
    float* dst = output->buffer().as<float*>();

    size_t dst_i = 0;
    for (size_t i0 = 0; i0 < out_dims[0]; ++i0) {
        for (size_t i1 = 0; i1 < out_dims[1]; ++i1) {
            for (size_t i2 = 0; i2 < out_dims[2]; ++i2) {
                for (size_t i3 = 0; i3 < out_dims[3]; ++i3) {
                    size_t src_i = in_strides[order[0]] * i0 +
                        in_strides[order[1]] * i1 +
                        in_strides[order[2]] * i2 +
                        in_strides[order[3]] * i3;

                    dst[dst_i++] = src[src_i];
                }
            }
        }
    }
}

// TODO: Replace with ngraph::op::v1::Transpose::evaluate when it will be available
BlobVector refPermute(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto input = toDefLayout(toFP32(inputs.at(0)));
    const auto order_blob = inputs.at(1);

    IE_ASSERT(order_blob->getTensorDesc().getPrecision() == Precision::I64);
    IE_ASSERT(order_blob->size() == 4u);

    int64_t* order = order_blob->buffer();

    const auto& dims = input->getTensorDesc().getDims();
    SizeVector new_dims;
    for (size_t i = 0; i < order_blob->size(); ++i) {
        new_dims.push_back(dims[order[i]]);
    }

    const auto  desc = TensorDesc(Precision::FP32, new_dims, Layout::NCHW);
    auto  output = make_blob_with_precision(desc);
    output->allocate();

    transpose(input, order, output);

    return {output};
};

} // namespace

TestNetwork& PermuteLayerDef::build() {
    const auto node  = std::make_shared<ngraph::op::Transpose>(testNet.getPort(inputPort),
                                                               testNet.getPort(constPort));

    return testNet.addLayer(name, node, refPermute);
}
