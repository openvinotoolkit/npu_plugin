//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

#include <ngraph/op/util/attr_types.hpp>

struct MVNParams final {
    bool normalize_variance = true;
    bool across_channels = true;
    float eps = 0.001f;
};

struct MVNLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo inputPort;
    MVNParams params;

    MVNLayerDef(TestNetwork& testNet, std::string name, MVNParams params)
            : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    MVNLayerDef& input(const std::string& lName, size_t port = 0) {
        inputPort = PortInfo(lName, port);
        return *this;
    }

    TestNetwork& build();
};
