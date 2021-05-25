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

struct ScaleShiftLayerDef final {
    TestNetwork& testNet;

    std::string name;

    PortInfo inputPort;
    PortInfo scalePort;
    PortInfo shiftPort;

    ScaleShiftLayerDef(TestNetwork& testNet, std::string name)
        : testNet(testNet), name(std::move(name)) {
    }

    ScaleShiftLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    ScaleShiftLayerDef& scale(const std::string& layerName, size_t index = 0) {
        scalePort = PortInfo(layerName, index);
        return *this;
    }
    ScaleShiftLayerDef& scale(const Blob::Ptr& blob) {
        const auto scaleLayerName = name + "_scale";
        testNet.addConst(scaleLayerName, blob);
        return scale(scaleLayerName);
    }
    ScaleShiftLayerDef& scale(float val, const Precision& precision, size_t numDims) {
        const auto scaleBlob = vpux::makeScalarBlob(val, precision, numDims);
        return scale(scaleBlob);
    }

    ScaleShiftLayerDef& shift(const std::string& layerName, size_t index = 0) {
        shiftPort = PortInfo(layerName, index);
        return *this;
    }
    ScaleShiftLayerDef& shift(const Blob::Ptr& blob) {
        const auto shiftLayerName = name + "_shift";
        testNet.addConst(shiftLayerName, blob);
        return shift(shiftLayerName);
    }
    ScaleShiftLayerDef& shift(float val, const Precision& precision, size_t numDims) {
        const auto shiftBlob = vpux::makeScalarBlob(val, precision, numDims);
        return shift(shiftBlob);
    }

    TestNetwork& build();
};
