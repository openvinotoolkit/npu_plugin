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

#pragma once

#include "vpux/compiler/core/dim.hpp"
#include "vpux/compiler/core/shape.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {

//
// NetworkInformation
//

mlir::LogicalResult verifyNetworkInformation(mlir::Operation* op);

template <class ConcreteOp>
class NetworkInformation
        : public mlir::OpTrait::TraitBase<ConcreteOp, NetworkInformation> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyNetworkInformation(op);
    }
};

template <class NetworkInfoOp, class DataInfoOp, class EndOp>
mlir::LogicalResult checkNetworkDataInfoBlock(NetworkInfoOp op,
                                              mlir::Block::OpListType& block,
                                              StringRef blockName) {
    for (auto&& p : block | indexed) {
        auto& infoOp = p.value();

        if (static_cast<size_t>(p.index()) == block.size() - 1) {
            if (!mlir::isa<EndOp>(infoOp)) {
                return printTo(op.emitError(),
                               "Got wrong item #{0} in '{1}' {2} ('{3}'), "
                               "expected '{4}'",
                               p.index(),
                               NetworkInfoOp::getOperationName(),
                               blockName,
                               infoOp,
                               EndOp::getOperationName());
            }
        } else {
            if (!mlir::isa<DataInfoOp>(infoOp)) {
                return printTo(op.emitError(),
                               "Got wrong item #{0} in '{1}' {2} ('{3}'), "
                               "expected '{4}'",
                               p.index(),
                               NetworkInfoOp::getOperationName(),
                               blockName,
                               infoOp,
                               DataInfoOp::getOperationName());
            }
        }
    }

    return mlir::success();
}

//
// SoftMaxLayerInterface
//

mlir::LogicalResult verifySoftMaxLayer(mlir::Operation* op);

}  // namespace IE
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.hpp.inc>
