//
// Copyright 2019 Intel Corporation.
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

struct AddLayerDef final {
    TestNetwork& testNet;

    std::string name;

    PortInfo input1Port;
    PortInfo input2Port;

    ngraph::op::AutoBroadcastSpec broadcastSpec;

    AddLayerDef(TestNetwork& testNet, std::string name,
                const ngraph::op::AutoBroadcastSpec& broadcastSpec
                    = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))
        : testNet(testNet), name(std::move(name)), broadcastSpec(broadcastSpec) {
    }

    AddLayerDef& input1(const std::string& layerName, size_t index = 0) {
        input1Port = PortInfo(layerName, index);
        return *this;
    }

    AddLayerDef& input2(const std::string& layerName, size_t index = 0) {
        input2Port = PortInfo(layerName, index);
        return *this;
    }

    TestNetwork& build();
};
