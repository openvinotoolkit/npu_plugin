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

struct StridedSliceParams final {
    std::vector<int64_t> begins;
    std::vector<int64_t> ends;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisAxisMask;
};

struct StridedSliceLayerDef final {
    TestNetwork& testNet;
    std::string name;
    PortInfo inputPort;
    PortInfo beginsPort;
    PortInfo endsPort;
    PortInfo stridesPort;
    StridedSliceParams params;

    StridedSliceLayerDef(TestNetwork& testNet, std::string name, StridedSliceParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    StridedSliceLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }
    StridedSliceLayerDef& begins(const std::string& layerName, size_t index = 0) {
        beginsPort = PortInfo(layerName, index);
        return *this;
    }
    StridedSliceLayerDef& begins(const Blob::Ptr& blob) {
        const auto blobName = name + "_begins";
        testNet.addConst(blobName, blob);
        return begins(blobName);
    }
    StridedSliceLayerDef& ends(const std::string& layerName, size_t index = 0) {
        endsPort = PortInfo(layerName, index);
        return *this;
    }
    StridedSliceLayerDef& ends(const Blob::Ptr& blob) {
        const auto blobName = name + "_ends";
        testNet.addConst(blobName, blob);
        return ends(blobName);
    }
    StridedSliceLayerDef& strides(const std::string& layerName, size_t index = 0) {
        stridesPort = PortInfo(layerName, index);
        return *this;
    }
    StridedSliceLayerDef& strides(const Blob::Ptr& blob) {
        const auto blobName = name + "_strides";
        testNet.addConst(blobName, blob);
        return strides(blobName);
    }

    TestNetwork& build();
};
