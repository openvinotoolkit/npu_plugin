//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct SoftmaxLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo inputPort;
    size_t axisSet;

    SoftmaxLayerDef(TestNetwork& testNet, std::string name, const size_t& axis_set)
            : testNet(testNet), name(std::move(name)), axisSet(axis_set) {
    }

    SoftmaxLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    TestNetwork& build();
};
