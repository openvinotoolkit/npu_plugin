//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/dma.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

int64_t vpux::getDMAPortValue(mlir::Operation* wrappedTaskOp) {
    if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(wrappedTaskOp)) {
        auto portAttr = dmaOp.getPortAttribute();
        if (portAttr == nullptr) {
            return 0;
        }
        return portAttr.getInt();
    }

    VPUX_THROW("Could not cast to DMA task '{0}'", *wrappedTaskOp);
}

VPUIP::DmaChannelType vpux::setDMAChannelType(VPUIP::DMATypeOpInterface /* dmaOp */, VPU::ArchKind /* arch */) {
    return VPUIP::DmaChannelType::DDR;
}
