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

#include "test_model/kmb_test_model.hpp"
#include "test_model/kmb_test_utils.hpp"

struct ReshapeLayerDef final {
    TestNetwork& net_;
    std::string  name_;

    PortInfo in_port_;
    PortInfo shape_port_;

    ReshapeLayerDef(TestNetwork& net, std::string name)
        : net_(net), name_(std::move(name)) {
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
