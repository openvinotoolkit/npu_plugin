//
// Copyright 2019 Intel Corporation.
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

struct TopKParams final {
    TopKParams(size_t axis, ngraph::op::v1::TopK::Mode modeType, ngraph::op::v1::TopK::SortType sortType)
        : _axis(axis), _modeType(modeType), _sortType(sortType) {}

    size_t _axis;
    ngraph::op::v1::TopK::Mode _modeType;
    ngraph::op::v1::TopK::SortType _sortType;
};

namespace llvm {

template <>
struct format_provider<ngraph::op::v1::TopK::Mode> final {
    static void format(const ngraph::op::v1::TopK::Mode& mode, llvm::raw_ostream& stream, StringRef style) {
        switch (mode) {
            case ngraph::op::v1::TopK::Mode::MAX:
            stream << "MAX";
            return;
            case ngraph::op::v1::TopK::Mode::MIN:
            stream << "MIN";
            return;
        }
    }
};

template <>
struct format_provider<ngraph::op::v1::TopK::SortType> final {
    static void format(const ngraph::op::v1::TopK::SortType& st, llvm::raw_ostream& stream, StringRef style) {
        switch(st) {
            case ngraph::op::v1::TopK::SortType::SORT_INDICES:
            stream << "Sort indices";
            return;
            case ngraph::op::v1::TopK::SortType::SORT_VALUES:
            stream << "Sort values";
            return;
            default:
            case ngraph::op::v1::TopK::SortType::NONE:
            stream << "<NONE>";
            return;
        }
    }
};
}

template <typename Stream>
inline Stream& operator<<(Stream& os, const TopKParams& p) {
    vpux::printTo(os, "[_axis:%v,_mode:%v,_sort:%v]", p._axis, p._modeType, p._sortType);
    return os;
}

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
