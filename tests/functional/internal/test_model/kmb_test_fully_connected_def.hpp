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
