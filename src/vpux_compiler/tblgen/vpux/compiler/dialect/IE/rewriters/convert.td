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
