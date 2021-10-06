//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/compiler/act_kernels/act_kernel_gen.h"

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/movitools/movitools.h"

using namespace vpux;
using namespace mlir;

namespace vpux {
namespace VPUIP {

VPUIP::BlobWriter::SpecificTask ACTShaveTaskOp::serialize(VPUIP::BlobWriter& ) {
    //return writer.createACTShaveTask(*this);
    return {};
}

VPUIP::BlobWriter::SpecificTask  SW_Kernel::serialize(vpux::VPUIP::BlobWriter& writer) {
    return writer.createSW_KernelTask(*this);
}

 mlir::ParseResult SW_Kernel::parseIOForward(mlir::OpAsmParser& parser,
                                        mlir::SmallVectorImpl<mlir::OpAsmParser::OperandType> &args,
                                        /*mlir::SmallVectorImpl<mlir::OpAsmParser::OperandType> &innerArgs,*/
                                        mlir::SmallVectorImpl<mlir::Type> &argsTypes) {
     if (parser.parseLParen())
         return ::mlir::failure();


     OpAsmParser::OperandType outerArg;
     if (parser.parseOperand(outerArg))
         return ::mlir::failure();

     args.push_back(outerArg);

     if (parser.parseColon())
         return ::mlir::failure();

     Type outerArgType;
     if (parser.parseType(outerArgType))
         return ::mlir::failure();

     argsTypes.push_back(outerArgType);

     if (parser.parseKeyword("as"))
         return ::mlir::failure();

     OpAsmParser::OperandType innerArg;
     if (parser.parseOperand(innerArg))
         return ::mlir::failure();

     //innerArgs.push_back(innerArg);

     if (parser.parseRParen())
         return ::mlir::failure();

     return {};
}

 void SW_Kernel::printIOForward(mlir::OpAsmPrinter& /*printer*/,
                           vpux::VPUIP::SW_Kernel&,
                           mlir::OperandRange /*args*/,
                           mlir::OperandRange::type_range /*argsTypes*/) {

 }
/*

mlir::ParseResult SW_Kernel_run::parseSW_Kernel_run(mlir::OpAsmParser& parser,
                                                    mlir::OperationState &*/
/*result*//*
) {
    //if (parser.parseLParen())
      //  return ::mlir::failure();

    llvm::SmallVector<OpAsmParser::OperandType> innerArg;

    if (parser.parseOperandList(innerArg, OpAsmParser::Delimiter::Paren))
        return ::mlir::failure();

     return {};
 }

 void SW_Kernel_run::print(mlir::OpAsmPrinter& */
/*printer*//*
, SW_Kernel_run &*/
/*runner*//*
){

 }
*/


}  // namespace VPUIP
}  // namespace vpux