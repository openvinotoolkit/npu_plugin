//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"
#include "sample_ext.hpp"
#include "add_with_offset_op.hpp"

#include <ngraph/op/util/attr_types.hpp>

struct AddWOffsetParams final {
    float offset = 1.1f;
};

struct AddWOffsetLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo input1Port;
    PortInfo input2Port;
    AddWOffsetParams params;

    AddWOffsetLayerDef(TestNetwork& testNet, std::string name, AddWOffsetParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    AddWOffsetLayerDef& input(const std::string& lName1, const std::string& lName2, size_t port = 0) {
        input1Port = PortInfo(lName1, port);
        input2Port = PortInfo(lName2, port);
        return *this;
    }

    TestNetwork& build();
};
