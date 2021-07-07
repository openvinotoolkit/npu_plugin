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

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {

//
// LayerInfoDialectInterface
//

class LayerInfoDialectInterface : public mlir::DialectInterface::Base<LayerInfoDialectInterface> {
public:
    explicit LayerInfoDialectInterface(mlir::Dialect* dialect): Base(dialect) {
    }

    virtual bool isSupportedPostProcessing(mlir::Operation* origOp, mlir::Operation* postOp) const = 0;
    virtual bool needToExpandChannels(mlir::Operation* origOp) const = 0;
    virtual bool isSupportedLayout(mlir::Operation* origOp, DataOrderInfo& info) const = 0;
};

}  // namespace IE
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.hpp.inc>
