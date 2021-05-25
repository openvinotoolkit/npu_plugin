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

struct SoftmaxLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo inputPort;
    size_t axisSet;

    SoftmaxLayerDef(TestNetwork& testNet, std::string name,
    const size_t& axis_set) : testNet(testNet), name(std::move(name)), axisSet(axis_set) {}

    SoftmaxLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    TestNetwork& build();
};
