//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/attributes/structs.hpp.inc>

namespace vpux {
namespace IERT {

//
// MemRefAttrLayout
//

class MemRefAttrLayout final :
        public mlir::MemRefLayoutAttrInterface::ExternalModel<MemRefAttrLayout, IERT::MemRefAttr> {
public:
    mlir::AffineMap getAffineMap(mlir::Attribute attr) const;

    bool isIdentity(mlir::Attribute) const;

    mlir::LogicalResult verifyLayout(mlir::Attribute attr, ArrayRef<int64_t> shape,
                                     FuncRef<mlir::InFlightDiagnostic()> emitError) const;
};

}  // namespace IERT
}  // namespace vpux
