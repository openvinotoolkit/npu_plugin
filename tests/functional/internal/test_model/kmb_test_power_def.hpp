//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "kmb_test_model.hpp"

struct PowerLayerDef final {
    TestNetwork& testNet;

    std::string name;

    PortInfo input1Port;
    PortInfo input2Port;

    ngraph::op::AutoBroadcastSpec broadcastSpec;

    PowerLayerDef(TestNetwork& testNet, std::string name,
                const ngraph::op::AutoBroadcastSpec& broadcastSpec
                    = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))
        : testNet(testNet), name(std::move(name)), broadcastSpec(broadcastSpec) {
    }

    PowerLayerDef& input1(const std::string& layerName, size_t index = 0) {
        input1Port = PortInfo(layerName, index);
        return *this;
    }
    PowerLayerDef& input1(const Blob::Ptr& blob) {
        const auto input1LayerName = name + "_input1";
        testNet.addConst(input1LayerName, blob);
        return input1(input1LayerName);
    }
    PowerLayerDef& input1(float val, const Precision& precision, size_t numDims) {
        const auto input1Blob = vpux::makeScalarBlob(val, precision, numDims);
        return input1(input1Blob);
    }

    PowerLayerDef& input2(const std::string& layerName, size_t index = 0) {
        input2Port = PortInfo(layerName, index);
        return *this;
    }
    PowerLayerDef& input2(const Blob::Ptr& blob) {
        const auto input2LayerName = name + "_input2";
        testNet.addConst(input2LayerName, blob);
        return input2(input2LayerName);
    }
    PowerLayerDef& input2(float val, const Precision& precision, size_t numDims) {
        const auto input2Blob = vpux::makeScalarBlob(val, precision, numDims);
        return input2(input2Blob);
    }

    TestNetwork& build();
};
