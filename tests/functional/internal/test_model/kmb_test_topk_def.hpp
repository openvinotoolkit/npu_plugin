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
#include "kmb_test_utils.hpp"

struct TopKParams final {
    TopKParams(size_t axis, ngraph::op::v1::TopK::Mode modeType, ngraph::op::v1::TopK::SortType sortType)
        : _axis(axis), _modeType(modeType), _sortType(sortType) {}

    size_t _axis;
    ngraph::op::v1::TopK::Mode _modeType;
    ngraph::op::v1::TopK::SortType _sortType;
};

std::ostream& operator<<(std::ostream& os, const TopKParams& p);

struct TopKLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo inputPort;
    PortInfo scalarKPort;

    TopKParams params;

    TopKLayerDef(TestNetwork& testNet, std::string name, TopKParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {}

    TopKLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    TopKLayerDef& scalarK(const std::string& layerName, size_t index = 0) {
        scalarKPort = PortInfo(layerName, index);
        return *this;
    }
    TopKLayerDef& scalarK(const Blob::Ptr& blob) {
        const auto scalarKLayerName = name + "_scalark";
        testNet.addConst(scalarKLayerName, blob);
        return scalarK(scalarKLayerName);
    }

    TestNetwork& build();
};
