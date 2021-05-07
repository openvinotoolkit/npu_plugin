//
// Copyright 2021 Intel Corporation.
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
