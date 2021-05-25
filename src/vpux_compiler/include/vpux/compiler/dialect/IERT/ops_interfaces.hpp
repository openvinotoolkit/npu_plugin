//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"

#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace IERT {

//
// LayerInfoDialectInterface
//

class LayerInfoDialectInterface : public mlir::DialectInterface::Base<LayerInfoDialectInterface> {
public:
    explicit LayerInfoDialectInterface(mlir::Dialect* dialect): Base(dialect) {
    }

    virtual mlir::Attribute getExecutor(mlir::Operation* op, uint32_t& numUnits) const = 0;
    virtual mlir::LogicalResult isSupportedLayout(mlir::Operation* origOp, DataOrderInfo& info) const = 0;
};

}  // namespace IERT
}  // namespace vpux
