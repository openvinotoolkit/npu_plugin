//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT_COPY_TO_DMA
#define VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT_COPY_TO_DMA

include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"

//
// VPUIP.Copy -> VPUIP.NNDMA
//

def createNNDMAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::NNDMAOp>($_loc, $0, $1)
        }]>;

def createUPADMAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::UPADMAOp>($_loc, $0, $1)
        }]>;

def RewriteCopy :
        Pat<
            (VPUIP_CopyOp $input, $output),
            (createNNDMAOp $input, $output)
        >;

#endif
