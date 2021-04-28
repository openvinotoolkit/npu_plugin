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
