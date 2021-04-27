//
// Copyright Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_IERT_REWRITERS_CONVERT
#define VPUX_COMPILER_DIALECT_IERT_REWRITERS_CONVERT

include "vpux/compiler/dialect/IERT/ops.td"

include "mlir/IR/OpBase.td"

//
// FuseReorders
//

def NotABlockArgument: Constraint<CPred<"!$_self.isa<mlir::BlockArgument>()">>;

def FuseReorders :
        Pat<
            (IERT_ReorderOp (IERT_ReorderOp $input, $output_buff1), $output_buff2),
            (IERT_ReorderOp $input, $output_buff2),
            [
                (NotABlockArgument:$output_buff1)
            ]
        >;

#endif
