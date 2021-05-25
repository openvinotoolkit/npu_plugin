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

#include "kmb_test_convert_def.hpp"

namespace {

BlobVector refConvert(const TestNetwork::NodePtr&, const BlobVector& inputs, const TestNetwork&) {
    return inputs;
}

}  // namespace

TestNetwork& ConvertLayerDef::build() {
    std::shared_ptr<ngraph::Node> convertNode =
        std::make_shared<ngraph::op::v0::Convert>(
            testNet.getPort(inputPort), params.destination_type);

    return testNet.addLayer(name, convertNode, refConvert);
}
