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

#include "kmb_test_convert_def.hpp"

#include <precision_utils.h>

#include <blob_factory.hpp>
#include <ngraph/runtime/reference/convert.hpp>

#include "vpu/utils/ie_helpers.hpp"

namespace {

template <typename From, typename To>
void convert(const Blob::Ptr &input, Blob::Ptr &output, const std::function<To(From)>& convFunc) {
    IE_ASSERT(input->size() == output->size());
    const auto srcPtr = input->buffer().as<const From *>();
    const auto dstPtr = output->buffer().as<To *>();
    for (size_t i = 0; i < input->size(); i++) {
        dstPtr[i] = convFunc(srcPtr[i]);
    }
}

BlobVector refConvert(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto convertLayer = std::dynamic_pointer_cast<ngraph::op::v0::Convert>(layer);
    IE_ASSERT(convertLayer != nullptr);

    const auto dstPrecision = Precision::FP32;

    const auto input = inputs.at(0);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(dstPrecision, outDims, Layout::NHWC);
    auto output = make_blob_with_precision(outDesc);
    output->allocate();

    vpu::copyBlob(input, output);

    return {output};
}

}  // namespace

TestNetwork& ConvertLayerDef::build() {
    std::shared_ptr<ngraph::Node> convertNode =
        std::make_shared<ngraph::op::v0::Convert>(
            testNet.getPort(inputPort), params.destination_type);

    return testNet.addLayer(name, convertNode, refConvert);
}
