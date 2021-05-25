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

#include <ngraph/op/util/attr_types.hpp>

struct PReluParams final {};

struct PReluLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PReluParams params;

    PortInfo inputPort;
    PortInfo weightsPort;

    PReluLayerDef(TestNetwork& testNet, std::string name, PReluParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    PReluLayerDef& input(const std::string& lName, size_t port = 0) {
        inputPort = PortInfo(lName, port);
        return *this;
    }

    PReluLayerDef& weights(const std::string& lName, size_t port = 0) {
        weightsPort = PortInfo(lName, port);
        return *this;
    }

    PReluLayerDef& weights(const Blob::Ptr& blob) {
        const auto weightsLayerName = name + "_weights";
        testNet.addConst(weightsLayerName, blob);
        return weights(weightsLayerName);
    }

    TestNetwork& build();
};
