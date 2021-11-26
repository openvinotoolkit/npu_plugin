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

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include <unordered_map>

namespace vpux {
namespace VPUIP {

struct QuantInfo {
    double rMin;
    double rMax;
    double scale;
    int64_t zeroPoint;
    int64_t postShift;
};

struct PwlQuantReqs {
    QuantInfo input;
    QuantInfo output;
};

extern const std::unordered_map<VPUIP::PPELayerType, PwlQuantReqs> pwlQuantReqs;

PwlQuantReqs getPwlQuantReqs(const VPUIP::PPELayerType ppeType);

}  // namespace VPUIP
}  // namespace vpux
