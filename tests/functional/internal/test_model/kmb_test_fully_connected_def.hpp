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

#pragma once

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct FullyConnectedParams final {
    size_t _outChannels = 0;

    FullyConnectedParams &outChannels(const size_t &outChannels) {
        this->_outChannels = outChannels;
        return *this;
    }
};
inline std::ostream& operator<<(std::ostream& os, const FullyConnectedParams& p) {
    vpu::formatPrint(os, "[outChannels:%v]", p._outChannels);
    return os;
}

struct FullyConnectedLayerDef final {
    TestNetwork& testNet;

    std::string name;

    FullyConnectedParams params;

    PortInfo inputPort;
    PortInfo weightsPort;
    PortInfo biasesPort;

    FullyConnectedLayerDef(TestNetwork& testNet, std::string name, FullyConnectedParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    FullyConnectedLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    FullyConnectedLayerDef& weights(const std::string& layerName, size_t index = 0) {
        weightsPort = PortInfo(layerName, index);
        return *this;
    }
    FullyConnectedLayerDef& weights(const Blob::Ptr& blob) {
        const auto weightsLayerName = name + "_weights";
        testNet.addConst(weightsLayerName, blob);
        return weights(weightsLayerName);
    }

    FullyConnectedLayerDef& biases(const std::string& layerName, size_t index = 0) {
        biasesPort = {layerName, index};
        return *this;
    }
    FullyConnectedLayerDef& biases(const Blob::Ptr& blob) {
        const auto biasesLayerName = name + "_biases";
        testNet.addConst(biasesLayerName, blob);
        return biases(biasesLayerName);
    }

    TestNetwork& build();
};

TensorDesc getFCWeightsDesc(const FullyConnectedParams& params, size_t inChannels, Precision precision);
TensorDesc getFCBiasesDesc(const FullyConnectedParams& params, Precision precision);
