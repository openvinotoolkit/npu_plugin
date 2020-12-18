//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

//
// RunTimeLayer
//

void vpux::IERT::getLayerEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerEffects");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Got non Layer Operation '{0}' in getLayerEffects", op->getName());

    for (const auto input : layer.getInputs()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), input);
    }

    for (const auto output : layer.getOutputs()) {
        effects.emplace_back(mlir::MemoryEffects::Write::get(), output);
    }
}
