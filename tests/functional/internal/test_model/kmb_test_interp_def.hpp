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

struct InterpParams final {
    InterpParams(){}
    InterpParams(const size_t& aligh_corners, const size_t& antialias, const size_t& pad_beg, const size_t& pad_end ) 
    : _alignCorners(aligh_corners), _antialias(antialias), _padBeg(pad_beg), _padEnd(pad_end){}
    size_t _alignCorners;
    size_t _antialias;
    size_t _padBeg;
    size_t _padEnd;
};

std::ostream& operator<<(std::ostream& os, const InterpParams& p);

struct InterpLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo inputPort;
    PortInfo outshapePort;

    InterpParams params;

    InterpLayerDef(TestNetwork& testNet, std::string name, InterpParams params) : 
    testNet(testNet), name(std::move(name)), params(std::move(params)) {}

    InterpLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }
    
    InterpLayerDef& outshape(const std::string& lName, size_t index = 0) {
        outshapePort = PortInfo(lName, index);
        return *this;
    }

    InterpLayerDef& outshape(const Blob::Ptr& blob) {
        const auto scaleLayerName = name + "_outshape";
        testNet.addConst(scaleLayerName, blob);
        return outshape(scaleLayerName);
    }

    TestNetwork& build();
};
