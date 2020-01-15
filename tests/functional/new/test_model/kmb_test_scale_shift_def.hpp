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
        const auto scaleBlob = makeScalarBlob(val, precision, numDims);
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
        const auto shiftBlob = makeScalarBlob(val, precision, numDims);
        return shift(shiftBlob);
    }

    TestNetwork& build();
};
