//
// Copyright Intel Corporation.
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

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace IE {

//
// IELayer
//

mlir::LogicalResult verifyIELayerOp(mlir::Operation* op);

template <typename ConcreteOp>
class IELayer : public mlir::OpTrait::TraitBase<ConcreteOp, IELayer> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyIELayerOp(op);
    }
};

}  // namespace IE
}  // namespace vpux
