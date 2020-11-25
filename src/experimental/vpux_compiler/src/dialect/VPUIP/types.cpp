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

#include "vpux/compiler/dialect/VPUIP/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// BarrierType
//

mlir::Type vpux::VPUIP::BarrierType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& /*parser*/) {
    return BarrierType::get(ctxt);
}

void vpux::VPUIP::BarrierType::print(mlir::DialectAsmPrinter& printer) const {
    printer << BarrierType::getMnemonic();
}

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES
