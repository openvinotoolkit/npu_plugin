//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

//
// DPUTaskOp
//

void vpux::VPUIP::DPUTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr start,
                                   mlir::ArrayAttr end, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                   vpux::VPUIP::MPEMode mpe_mode) {
    build(builder, state, start, end, pads_begin, pads_end, mpe_mode, mlir::ValueRange{}, mlir::ValueRange{});
}
