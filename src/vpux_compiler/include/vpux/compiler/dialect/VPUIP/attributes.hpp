//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/attr_interfaces.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/attributes.hpp.inc>

#include <vpux/compiler/dialect/VPUIP/structs.hpp.inc>
namespace vpux {
namespace VPUIP {

//
// MemRefAttrLayout
//

class MemRefAttrLayout final :
        public mlir::MemRefLayoutAttrInterface::ExternalModel<MemRefAttrLayout, VPUIP::MemRefAttr> {
public:
    using ConcreteEntity = mlir::DictionaryAttr;

    mlir::AffineMap getAffineMap(mlir::Attribute attr) const;

    bool isIdentity(mlir::Attribute) const;

    mlir::LogicalResult verifyLayout(mlir::Attribute attr, ArrayRef<int64_t> shape,
                                     FuncRef<mlir::InFlightDiagnostic()> emitError) const;
};

//
// CompressionSchemeAttr
//

VPUIP::CompressionSchemeAttr getCompressionSchemeAttr(mlir::Type type);
mlir::Type setCompressionSchemeAttr(mlir::Type type, VPUIP::CompressionSchemeAttr compressionSchemeAttr);

VPUIP::CompressionSchemeAttr tileCompressionScheme(VPUIP::CompressionSchemeAttr compressionScheme, ShapeRef tileOffsets,
                                                   ShapeRef tileShape);
mlir::Type tileTypeCompressionScheme(mlir::Type type, ShapeRef tileOffsets, ShapeRef tileShape);

}  // namespace VPUIP
}  // namespace vpux
