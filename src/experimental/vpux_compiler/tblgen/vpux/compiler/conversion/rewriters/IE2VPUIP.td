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

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_IE2VPUIP
#define VPUX_COMPILER_CONVERSION_REWRITERS_IE2VPUIP

include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"

//
// CopyOp
//

def NullAttr : Attr<CPred<"$_self == nullptr">>;

def createUPADMAOp : NativeCodeCall<"$_builder.create<vpux::VPUIP::UPADMAOp>($_loc, $0, $1)">;

def RewriteCopyOp :
        Pat<
            (CopyOp AnyMemRef:$src, AnyMemRef:$dst, NullAttr:$inputPermutation, NullAttr:$outputPermutation),
            (createUPADMAOp $src, $dst)
        >;

#endif
