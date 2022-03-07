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

#include <openvino/itt.hpp>

namespace vpux {
namespace itt {
namespace domains {

OV_ITT_DOMAIN(VPUXPlugin);
OV_ITT_DOMAIN(Hddl2Backend);
OV_ITT_DOMAIN(VpualBackend);
OV_ITT_DOMAIN(LevelZeroBackend);
OV_ITT_DOMAIN(ImdBackend);
OV_ITT_DOMAIN(EmulatorBackend);

}  // namespace domains
}  // namespace itt
}  // namespace vpux
