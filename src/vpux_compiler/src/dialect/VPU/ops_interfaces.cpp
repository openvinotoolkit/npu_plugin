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

#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"

using namespace vpux;

//
// SparseOpInterface
//

bool vpux::VPU::supportsSparseInputs(VPU::SparseOpInterface op) {
    return (op.sparsitySupport() & VPU::SparsitySupport::SPARSE_INPUTS) != VPU::SparsitySupport::NONE;
}

bool vpux::VPU::supportsSparseOutputs(VPU::SparseOpInterface op) {
    return (op.sparsitySupport() & VPU::SparsitySupport::SPARSE_OUTPUTS) != VPU::SparsitySupport::NONE;
}

bool vpux::VPU::supportsSparseData(VPU::SparseOpInterface op) {
    return supportsSparseInputs(op) && supportsSparseOutputs(op);
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.cpp.inc>
