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
