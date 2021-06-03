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

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_VIEW_OPS_TO_VPUIP
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_VIEW_OPS_TO_VPUIP

include "vpux/compiler/dialect/IERT/ops.td"
include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"

//
// IERT.ConcatView -> output_buff
//

def RewriteConcatView :
        Pat<
            (IERT_ConcatViewOp $inputs, $output_buff),
            (replaceWithValue $output_buff)
        >;

#endif
