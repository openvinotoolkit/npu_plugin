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

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/utils/asm.hpp"

//
// Generated
//
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/generated/ops.cpp.inc>

namespace vpux {
namespace VPU {
SmallVector<Byte> getMemSizes(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, SmallVector<Byte>>(op)
            .Case<NCEConvolutionOp>([](NCEConvolutionOp origOp) {
                return origOp.memSizes(origOp.strides(), origOp.input().getType().cast<vpux::NDTypeInterface>(),
                                       origOp.filter().getType().cast<vpux::NDTypeInterface>(),
                                       origOp.output().getType().cast<vpux::NDTypeInterface>());
            })
            .Case<NCEMaxPoolOp>([](NCEMaxPoolOp origOp) {
                return origOp.memSizes(origOp.kernel_size(), origOp.strides(),
                                       origOp.input().getType().cast<vpux::NDTypeInterface>(),
                                       origOp.output().getType().cast<vpux::NDTypeInterface>());
            })
            .Case<NCEEltwiseOp>([](NCEEltwiseOp origOp) {
                return origOp.memSizes(origOp.input1().getType().cast<vpux::NDTypeInterface>(),
                                       origOp.input2().getType().cast<vpux::NDTypeInterface>(),
                                       origOp.output().getType().cast<vpux::NDTypeInterface>());
            })
            .Case<NCEDepthConvolutionOp>([](NCEDepthConvolutionOp origOp) {
                return origOp.memSizes(origOp.strides(), origOp.input().getType().cast<vpux::NDTypeInterface>(),
                                       origOp.filter().getType().cast<vpux::NDTypeInterface>(),
                                       origOp.output().getType().cast<vpux::NDTypeInterface>());
            })
            .Default([](mlir::Operation* unknownOp) -> SmallVector<Byte> {
                VPUX_THROW("Operation CMX check '{0}' at '{1}' is not implemented", unknownOp->getName(),
                           unknownOp->getLoc());
            });
}
}  // namespace VPU
}  // namespace vpux
