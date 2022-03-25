//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_IE_REWRITERS_REORDER
#define VPUX_COMPILER_DIALECT_IE_REWRITERS_REORDER

include "vpux/compiler/dialect/IE/ops.td"

include "mlir/IR/OpBase.td"

//
// FuseReorders
//

def FuseReorders :
        Pat<
            (IE_ReorderOp (IE_ReorderOp $input, $dstOrder1), $dstOrder2),
            (IE_ReorderOp $input, $dstOrder2)
        >;

#endif
