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

struct NormalizeParams final {
    NormalizeParams(double eps, ngraph::op::EpsMode eps_mode) : _eps(eps), _eps_mode(eps_mode){}

    double _eps;
    ngraph::op::EpsMode _eps_mode;
};

std::ostream& operator<<(std::ostream& os, const NormalizeParams& p);

struct NormalizeLayerDef final {
    TestNetwork& testNet;

    std::string name;

    NormalizeParams params;

    PortInfo inputPort;
    PortInfo axesPort;

    NormalizeLayerDef(TestNetwork& testNet, std::string name, NormalizeParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    NormalizeLayerDef& input(const std::string& lName, size_t index = 0) {
        inputPort = PortInfo(lName, index);
        return *this;
    }
    NormalizeLayerDef& axes(const std::string& lName, size_t index = 0) {
        axesPort = PortInfo(lName, index);
        return *this;
    }
    NormalizeLayerDef& axes(const Blob::Ptr& blob) {
        const auto scaleLayerName = name + "_axes";
        testNet.addConst(scaleLayerName, blob);
        return axes(scaleLayerName);
    }
    TestNetwork& build();
};
