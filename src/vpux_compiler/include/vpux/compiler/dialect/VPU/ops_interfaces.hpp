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

#include "vpux/compiler/dialect/VPU/attributes.hpp"

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.hpp.inc>

namespace vpux {
namespace VPU {

//
// SparseOpInterface
//

bool supportsSparseInputs(VPU::SparseOpInterface op);
bool supportsSparseOutputs(VPU::SparseOpInterface op);
bool supportsSparseData(VPU::SparseOpInterface op);

}  // namespace VPU
}  // namespace vpux
