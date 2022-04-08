//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "test_model/kmb_test_model.hpp"
#include "test_model/kmb_test_utils.hpp"

struct ReshapeLayerDef final {
    TestNetwork& net_;
    std::string name_;

    PortInfo in_port_;
    PortInfo shape_port_;

    ReshapeLayerDef(TestNetwork& net, std::string name): net_(net), name_(std::move(name)) {
    }

    ReshapeLayerDef& input(const std::string& layer_name, size_t index = 0) {
        in_port_ = PortInfo(layer_name, index);
        return *this;
    }

    ReshapeLayerDef& shape(const std::string& layer_name, size_t index = 0) {
        shape_port_ = PortInfo(layer_name, index);
        return *this;
    }

    TestNetwork& build();
};
