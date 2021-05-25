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

#include <ngraph/op/util/attr_types.hpp>

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct GatherLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo inputPort;
    PortInfo indicesPort;
    PortInfo axisPort;

    GatherLayerDef(TestNetwork& testNet, std::string name)
        : testNet(testNet), name(std::move(name)) {}

    GatherLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    GatherLayerDef& indices(const std::string& layerName, size_t index = 0) {
        indicesPort = PortInfo(layerName, index);
        return *this;
    }

    GatherLayerDef& axis(const std::string& layerName, size_t index = 0) {
        axisPort = PortInfo(layerName, index);
        return *this;
    }

    TestNetwork& build();
};
