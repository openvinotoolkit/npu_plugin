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

struct InterpParams final {
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
