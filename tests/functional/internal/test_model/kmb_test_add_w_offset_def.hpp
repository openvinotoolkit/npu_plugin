//
// Copyright 2021 Intel Corporation.
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
