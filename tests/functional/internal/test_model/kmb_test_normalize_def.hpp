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
