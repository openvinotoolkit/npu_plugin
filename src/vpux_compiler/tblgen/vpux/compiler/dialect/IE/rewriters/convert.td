//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_IE_REWRITERS_CONVERT
#define VPUX_COMPILER_DIALECT_IE_REWRITERS_CONVERT

include "vpux/compiler/dialect/IE/ops.td"

include "mlir/IR/OpBase.td"

//
// FuseConverts
//

def FuseConverts :
        Pat<
            (IE_ConvertOp (IE_ConvertOp $input, $dstType1), $dstType2),
            (IE_ConvertOp $input, $dstType2)
        >;

#endif
