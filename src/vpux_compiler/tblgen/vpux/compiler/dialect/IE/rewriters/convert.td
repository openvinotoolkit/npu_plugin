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
