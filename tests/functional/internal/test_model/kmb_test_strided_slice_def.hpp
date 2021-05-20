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
