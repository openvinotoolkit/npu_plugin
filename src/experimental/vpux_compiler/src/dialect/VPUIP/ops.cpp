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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/dims_order.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpImplementation.h>

using namespace vpux;

namespace {

class VPUIPDialectAsmHooks final : public mlir::OpAsmDialectInterface {
public:
    using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

public:
    mlir::LogicalResult getAlias(mlir::Attribute attr,
                                 llvm::raw_ostream& os) const final;
};

mlir::LogicalResult
        VPUIPDialectAsmHooks::getAlias(mlir::Attribute attr,
                                       llvm::raw_ostream& os) const {
    if (const auto affineMapAttr = attr.dyn_cast<mlir::AffineMapAttr>()) {
        if (const auto dimsOrder =
                    DimsOrder::fromAffineMap(affineMapAttr.getValue())) {
            if (const auto name = dimsOrder->getCanonicalName()) {
                os << name.getValue();
                return mlir::success();
            }
        }
    }

    return mlir::failure();
}

}  // namespace

void vpux::VPUIP::VPUIPDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
#undef GET_OP_LIST
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIP/generated/types.cpp.inc>
#undef GET_TYPEDEF_LIST
            >();

    addInterfaces<VPUIPDialectAsmHooks>();
}

mlir::Type vpux::VPUIP::VPUIPDialect::parseType(
        mlir::DialectAsmParser& parser) const {
    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()),
                "Failed to get VPUIP Type mnemonic");
        return nullptr;
    }

    const auto type = generatedTypeParser(getContext(), parser, mnemonic);

    if (type == nullptr) {
        printTo(parser.emitError(parser.getCurrentLocation()),
                "Unknown VPUIP Type '{0}'",
                mnemonic);
    }

    return type;
}

void vpux::VPUIP::VPUIPDialect::printType(mlir::Type type,
                                          mlir::DialectAsmPrinter& os) const {
    generatedTypePrinter(type, os);
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
#undef GET_OP_CLASSES
