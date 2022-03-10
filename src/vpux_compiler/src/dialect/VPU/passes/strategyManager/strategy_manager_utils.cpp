//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/VPU/strategy_manager_utils.hpp"
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

// This channel major convolution efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> vpux::VPU::channelMajorEfficiencyTable() {
    std::map<int64_t, std::map<int64_t, double>> table = {{
            {3, {{1, 0.253}, {2, 0.183594}, {4, 0.183594}}},
            {5, {{1, 0.535156}, {2, 0.2773}, {4, 0.152344}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}}},
    }};
    return table;
}

// This depthwise convolution efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> vpux::VPU::depthwiseEfficiencyTable() {
    std::map<int64_t, std::map<int64_t, double>> table = {{
            {3, {{1, 0.165}, {2, 0.128}, {4, 0.128}, {6, 0.165}}},
            {5, {{1, 0.483}, {2, 0.241}, {4, 0.132}, {6, 0.483}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}, {6, 0.0395}}},
            {9, {{1, 0.8008}, {2, 0.4687}, {4, 0.2266}, {6, 0.8008}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}, {6, 0.9023}}},
    }};
    return table;
}

double vpux::VPU::getChannelMajorEfficiencyConstant(int64_t kernel, int64_t stride) {
    if (channelMajorEfficiencyTable().count(kernel)) {
        auto table = channelMajorEfficiencyTable()[kernel];
        if (table.count(stride)) {
            return channelMajorEfficiencyTable()[kernel][stride];
        }
        VPUX_THROW("The stride size {0} does not exist in the channel major efficiency table", stride);
    }
    VPUX_THROW("The kernel size {0} does not exist in the channel major efficiency table", kernel);
}

double vpux::VPU::getDepthwiseEfficiencyConstant(int64_t kernel, int64_t stride) {
    if (depthwiseEfficiencyTable().count(kernel)) {
        auto table = depthwiseEfficiencyTable()[kernel];
        if (table.count(stride)) {
            return depthwiseEfficiencyTable()[kernel][stride];
        }
        VPUX_THROW("The stride size {0} does not exist in the depthwise efficiency table", stride);
    }
    VPUX_THROW("The kernel size {0} does not exist in the depthwise efficiency table", kernel);
}
