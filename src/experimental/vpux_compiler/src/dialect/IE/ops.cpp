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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

class IEDialectOpAsmHooks final : public mlir::OpAsmDialectInterface {
public:
    using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

public:
    void getAsmResultNames(mlir::Operation* op,
                           mlir::OpAsmSetValueNameFn setNameFn) const final;
};

void IEDialectOpAsmHooks::getAsmResultNames(
        mlir::Operation* op,
        mlir::OpAsmSetValueNameFn setNameFn) const {
    if (const auto nameLoc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
        setNameFn(op->getResult(0), nameLoc.getName());
    }
}

}  // namespace

void vpux::IE::IEDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
#undef GET_OP_LIST
            >();

    addAttributes<LayoutAttr>();

    addInterfaces<IEDialectOpAsmHooks>();
}

mlir::Attribute
        vpux::IE::IEDialect::parseAttribute(mlir::DialectAsmParser& parser,
                                            mlir::Type) const {
    StringRef mnenomic;
    if (mlir::failed(parser.parseKeyword(&mnenomic))) {
        printTo(parser.emitError(parser.getCurrentLocation()),
                "Failed to get IE Attribute mnenomic");
        return nullptr;
    }

    if (mnenomic == LayoutAttr::getMnemonic()) {
        return LayoutAttr::parse(parser);
    }

    printTo(parser.emitError(parser.getCurrentLocation()),
            "Unknown IE Attribute '{0}'",
            mnenomic);
    return nullptr;
}

void vpux::IE::IEDialect::printAttribute(mlir::Attribute attr,
                                         mlir::DialectAsmPrinter& os) const {
    llvm::TypeSwitch<mlir::Attribute>(attr).Case<LayoutAttr>(
            [&os](LayoutAttr layout) {
                layout.print(os);
            });
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
#undef GET_OP_CLASSES
