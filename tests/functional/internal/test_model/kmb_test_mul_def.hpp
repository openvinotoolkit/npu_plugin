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

#pragma once

#include "kmb_test_model.hpp"

struct MultiplyLayerDef final {
    TestNetwork& testNet;

    std::string name;

    PortInfo input1Port;
    PortInfo input2Port;

    ngraph::op::AutoBroadcastSpec broadcastSpec;

    MultiplyLayerDef(TestNetwork& testNet, std::string name,
                const ngraph::op::AutoBroadcastSpec& broadcastSpec
                    = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))
        : testNet(testNet), name(std::move(name)), broadcastSpec(broadcastSpec) {
    }

    MultiplyLayerDef& input1(const std::string& layerName, size_t index = 0) {
        input1Port = PortInfo(layerName, index);
        return *this;
    }
    MultiplyLayerDef& input1(const Blob::Ptr& blob) {
        const auto input1LayerName = name + "_input1";
        testNet.addConst(input1LayerName, blob);
        return input1(input1LayerName);
    }
    MultiplyLayerDef& input1(float val, const Precision& precision, size_t numDims) {
        const auto input1Blob = vpux::makeScalarBlob(val, precision, numDims);
        return input1(input1Blob);
    }

    MultiplyLayerDef& input2(const std::string& layerName, size_t index = 0) {
        input2Port = PortInfo(layerName, index);
        return *this;
    }
    MultiplyLayerDef& input2(const Blob::Ptr& blob) {
        const auto input2LayerName = name + "_input2";
        testNet.addConst(input2LayerName, blob);
        return input2(input2LayerName);
    }
    MultiplyLayerDef& input2(float val, const Precision& precision, size_t numDims) {
        const auto input2Blob = vpux::makeScalarBlob(val, precision, numDims);
        return input2(input2Blob);
    }

    TestNetwork& build();
};
