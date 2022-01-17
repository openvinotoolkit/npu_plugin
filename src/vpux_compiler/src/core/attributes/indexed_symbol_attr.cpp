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

#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux_compiler.hpp"

using namespace vpux;

IndexedSymbolAttr IndexedSymbolAttr::get(mlir::MLIRContext* context, mlir::ArrayRef<mlir::Attribute> array) {
    VPUX_THROW_UNLESS(!array.empty(), "Can not create empty indexed symbol attribute");

    if (array.size() == 1) {
        auto flatSymbolRef = array[0].dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
        VPUX_THROW_UNLESS(flatSymbolRef != nullptr, "Unsupported symbol attr type. Got {0}", array[0]);

        return flatSymbolRef.cast<IndexedSymbolAttr>();
    }

    auto arrayAttr = mlir::ArrayAttr::get(context, array);
    VPUX_THROW_UNLESS(IndexedSymbolAttr::classof(arrayAttr),
                      "Array content does not match with IndexedSymbolAttr type. Got {0}", array);

    return arrayAttr.cast<IndexedSymbolAttr>();
}

IndexedSymbolAttr IndexedSymbolAttr::get(mlir::MLIRContext* context, StringRef name) {
    return get(context, {mlir::FlatSymbolRefAttr::get(context, name)});
}

IndexedSymbolAttr IndexedSymbolAttr::get(mlir::StringAttr name) {
    return get(name.getContext(), {mlir::FlatSymbolRefAttr::get(name)});
}

mlir::FlatSymbolRefAttr IndexedSymbolAttr::getNameAttr() const {
    if (isa<mlir::FlatSymbolRefAttr>()) {
        return cast<mlir::FlatSymbolRefAttr>();
    }

    auto arrayAttr = cast<mlir::ArrayAttr>();
    return arrayAttr[0].dyn_cast<mlir::FlatSymbolRefAttr>();
}

mlir::Optional<mlir::IntegerAttr> IndexedSymbolAttr::getIndexAttr() const {
    if (isa<mlir::FlatSymbolRefAttr>()) {
        return mlir::None;
    }

    auto arrayAttr = cast<mlir::ArrayAttr>();
    if (arrayAttr.size() == 1) {
        return mlir::None;
    }

    if (auto idxAttr = arrayAttr[1].dyn_cast<mlir::IntegerAttr>()) {
        return idxAttr;
    }

    return mlir::None;
}

mlir::Optional<IndexedSymbolAttr> IndexedSymbolAttr::getNestedAttr() const {
    if (isa<mlir::FlatSymbolRefAttr>()) {
        return mlir::None;
    }

    auto arrayAttr = cast<mlir::ArrayAttr>();
    if (arrayAttr.size() == 1) {
        return mlir::None;
    }

    auto symIdx = arrayAttr.size() > 2 ? 2 : 1;
    if (auto symAttr = arrayAttr[symIdx].dyn_cast<IndexedSymbolAttr>()) {
        return symAttr;
    }

    return mlir::None;
}

StringRef IndexedSymbolAttr::getName() const {
    return getNameAttr().getValue();
}

bool IndexedSymbolAttr::isDefined() const {
    return getIndexAttr().hasValue();
}

int64_t IndexedSymbolAttr::getIndex() const {
    VPUX_THROW_UNLESS(isDefined(), "Attribute doesn't have index");

    return getIndexAttr()->getInt();
}

mlir::Attribute IndexedSymbolAttr::getValue() {
    return *this;
}

bool IndexedSymbolAttr::classof(mlir::Attribute attr) {
    if (attr.isa<mlir::FlatSymbolRefAttr>()) {
        return true;
    }

    auto indexedSym = attr.dyn_cast<mlir::ArrayAttr>();
    if (indexedSym == nullptr) {
        return false;
    }

    if (indexedSym.empty() || indexedSym.size() > 3) {
        return false;
    }

    if (!indexedSym[0].isa<mlir::FlatSymbolRefAttr>()) {
        return false;
    }

    if (indexedSym.size() == 1) {
        return true;
    }

    if (indexedSym.size() == 2) {
        return indexedSym[1].isa<mlir::IntegerAttr>() || indexedSym[1].isa<IndexedSymbolAttr>();
    }

    return indexedSym[1].isa<mlir::IntegerAttr>() && indexedSym[2].isa<IndexedSymbolAttr>();
}
