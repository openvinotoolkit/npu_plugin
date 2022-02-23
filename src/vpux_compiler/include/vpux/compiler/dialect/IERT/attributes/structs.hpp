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
