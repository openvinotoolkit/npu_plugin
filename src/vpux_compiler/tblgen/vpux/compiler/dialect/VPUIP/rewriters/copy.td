//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT
#define VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT

include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"

//
// FuseCopies
//

def NotABlockArgument: Constraint<CPred<"!$_self.isa<mlir::BlockArgument>()">>;

def FuseCopies :
        Pat<
            (VPUIP_CopyOp (VPUIP_CopyOp $input, $output_buff1), $output_buff2),
            (VPUIP_CopyOp $input, $output_buff2),
            [
                (NotABlockArgument:$output_buff1)
            ]
        >;

#endif
