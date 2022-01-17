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

#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

class IndexedSymbolAttr : public mlir::Attribute {
public:
    using mlir::Attribute::Attribute;

public:
    static IndexedSymbolAttr get(mlir::MLIRContext* context, mlir::ArrayRef<mlir::Attribute> array);
    static IndexedSymbolAttr get(mlir::MLIRContext* context, StringRef name);
    static IndexedSymbolAttr get(mlir::StringAttr name);

    mlir::FlatSymbolRefAttr getNameAttr() const;
    mlir::Optional<mlir::IntegerAttr> getIndexAttr() const;
    mlir::Optional<vpux::IndexedSymbolAttr> getNestedAttr() const;

    StringRef getName() const;

    bool isDefined() const;
    int64_t getIndex() const;

    mlir::Attribute getValue();

    static bool classof(mlir::Attribute attr);
};

}  // namespace vpux
