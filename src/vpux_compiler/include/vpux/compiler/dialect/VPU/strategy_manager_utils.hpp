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

#pragma once

#include <map>
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux {
namespace VPU {

std::map<int64_t, std::map<int64_t, double>> channelMajorEfficiencyTable();
std::map<int64_t, std::map<int64_t, double>> depthwiseEfficiencyTable();
double getChannelMajorEfficiencyConstant(int64_t kernel, int64_t stride);
double getDepthwiseEfficiencyConstant(int64_t kernel, int64_t stride);

}  // namespace VPU
}  // namespace vpux
