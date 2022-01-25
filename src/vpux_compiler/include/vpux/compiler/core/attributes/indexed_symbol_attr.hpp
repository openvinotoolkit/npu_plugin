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
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

class IndexedSymbolAttr : public mlir::Attribute {
public:
    using mlir::Attribute::Attribute;

public:
    static bool classof(mlir::Attribute attr);

public:
    static IndexedSymbolAttr get(mlir::MLIRContext* context, ArrayRef<mlir::Attribute> array);
    static IndexedSymbolAttr get(mlir::MLIRContext* context, StringRef name);
    static IndexedSymbolAttr get(mlir::StringAttr name);

public:
    Optional<IndexedSymbolAttr> getNestedReference() const;

public:
    mlir::FlatSymbolRefAttr getRootReference() const;
    mlir::StringAttr getRootNameAttr() const;
    StringRef getRootName() const;

    mlir::FlatSymbolRefAttr getLeafReference() const;
    mlir::StringAttr getLeafNameAttr() const;
    StringRef getLeafName() const;

    mlir::SymbolRefAttr getFullReference() const;

public:
    Optional<mlir::IntegerAttr> getIndexAttr() const;
    Optional<int64_t> getIndex() const;
};

}  // namespace vpux
