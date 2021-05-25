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
#include "kmb_test_utils.hpp"

struct FakeQuantizeLayerDef final {
    TestNetwork& testNet;

    std::string name;

    PortInfo inputPort;
    PortInfo inputLowPort;
    PortInfo inputHighPort;
    PortInfo outputLowPort;
    PortInfo outputHighPort;
    size_t levels;

    FakeQuantizeLayerDef(TestNetwork& testNet, std::string name, size_t levels)
        : testNet(testNet), name(std::move(name)), levels(levels) {
    }

    FakeQuantizeLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& input(const Blob::Ptr& blob) {
        const auto inputLayerName = name + "_input";
        testNet.addConst(inputLayerName, blob);
        return input(inputLayerName);
    }

    FakeQuantizeLayerDef& inputLow(const std::string& layerName, size_t index = 0) {
        inputLowPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& inputLow(const Blob::Ptr& blob) {
        const auto inputLowLayerName = name + "_inputLow";
        testNet.addConst(inputLowLayerName, blob);
        return inputLow(inputLowLayerName);
    }
    FakeQuantizeLayerDef& inputLow(float val, const Precision& precision) {
        const auto inputLowBlob = vpux::makeScalarBlob(val, precision);
        return inputLow(inputLowBlob);
    }

    FakeQuantizeLayerDef& inputHigh(const std::string& layerName, size_t index = 0) {
        inputHighPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& inputHigh(const Blob::Ptr& blob) {
        const auto inputHighLayerName = name + "_inputHigh";
        testNet.addConst(inputHighLayerName, blob);
        return inputHigh(inputHighLayerName);
    }
    FakeQuantizeLayerDef& inputHigh(float val, const Precision& precision) {
        const auto inputHighBlob = vpux::makeScalarBlob(val, precision);
        return inputHigh(inputHighBlob);
    }

    FakeQuantizeLayerDef& outputLow(const std::string& layerName, size_t index = 0) {
        outputLowPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& outputLow(const Blob::Ptr& blob) {
        const auto outputLowLayerName = name + "_outputLow";
        testNet.addConst(outputLowLayerName, blob);
        return outputLow(outputLowLayerName);
    }
    FakeQuantizeLayerDef& outputLow(float val, const Precision& precision) {
        const auto outputLowBlob = vpux::makeScalarBlob(val, precision);
        return outputLow(outputLowBlob);
    }

    FakeQuantizeLayerDef& outputHigh(const std::string& layerName, size_t index = 0) {
        outputHighPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& outputHigh(const Blob::Ptr& blob) {
        const auto outputHighLayerName = name + "_outputHigh";
        testNet.addConst(outputHighLayerName, blob);
        return outputHigh(outputHighLayerName);
    }
    FakeQuantizeLayerDef& outputHigh(float val, const Precision& precision) {
        const auto outputHighBlob = vpux::makeScalarBlob(val, precision);
        return outputHigh(outputHighBlob);
    }

    FakeQuantizeLayerDef& low(const std::string& layerName, size_t index = 0) {
        inputLow(layerName, index);
        outputLow(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& low(const Blob::Ptr& blob) {
        const auto lowLayerName = name + "_low";
        testNet.addConst(lowLayerName, blob);
        inputLow(lowLayerName);
        outputLow(lowLayerName);
        return *this;
    }
    FakeQuantizeLayerDef& low(float val, const Precision& precision) {
        const auto lowBlob = vpux::makeScalarBlob(val, precision);
        inputLow(lowBlob);
        outputLow(lowBlob);
        return *this;
    }

    FakeQuantizeLayerDef& high(const std::string& layerName, size_t index = 0) {
        inputHigh(layerName, index);
        outputHigh(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& high(const Blob::Ptr& blob) {
        const auto highLayerName = name + "_high";
        testNet.addConst(highLayerName, blob);
        inputHigh(highLayerName);
        outputHigh(highLayerName);
        return *this;
    }
    FakeQuantizeLayerDef& high(float val, const Precision& precision) {
        const auto highBlob = vpux::makeScalarBlob(val, precision);
        inputHigh(highBlob);
        outputHigh(highBlob);
        return *this;
    }

    TestNetwork& build();
};
