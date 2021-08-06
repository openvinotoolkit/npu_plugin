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

#ifndef VPUX_COMPILER_CORE_ATTRIBUTES
#define VPUX_COMPILER_CORE_ATTRIBUTES

include "mlir/IR/OpBase.td"

def IntAttr : Attr<CPred<"$_self.isa<mlir::IntegerAttr>()">, "Integer attribute"> {
    let storageType = "mlir::IntegerAttr";
    let valueType = I64;
    let returnType = "int64_t";
    let convertFromStorage = "$_self.getValue().getSExtValue()";
    let constBuilderCall = "$_builder.getI64IntegerAttr($0)";
}

#endif
