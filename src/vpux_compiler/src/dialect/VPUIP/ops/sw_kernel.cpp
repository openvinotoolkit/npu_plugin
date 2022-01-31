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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace mlir;

namespace vpux {
namespace VPUIP {

VPUIP::BlobWriter::SpecificTask SwKernelOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    return writer.createSW_KernelTask(*this);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex) {
    // looks this is a result types
    build(builder, opState, results.getTypes(), kernelFunction, inputs, results, tileIndex);
}

mlir::LogicalResult SwKernelOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                 mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                 mlir::RegionRange /*regions*/,
                                                 mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPUIP::SwKernelOpAdaptor swKernelOp(operands, attrs);
    if (mlir::failed(swKernelOp.verify(loc))) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(swKernelOp.inputs().size() == 1, "For now act-kernels with only one input are supported. Got {0}",
                      swKernelOp.inputs().size());
    VPUX_THROW_UNLESS(swKernelOp.output_buffs().size() == 1,
                      "For now act-kernels with only one output are supported. Got {0}",
                      swKernelOp.output_buffs().size());

    const auto inType = swKernelOp.inputs()[0].getType();
    const auto outType = swKernelOp.output_buffs()[0].getType();

    VPUX_THROW_UNLESS(inType == outType, "Operands of different type not yet supported: {0} vs {1}", inType, outType);

    inferredTypes.push_back(inType);

    return mlir::success();
}

IERT::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    return llvm::TypeSwitch<mlir::Operation*, IERT::KernelInfo>(origOp)
            .Case<IERT::ExpOp>([&](IERT::ExpOp) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"exp_fp16"}, {"exp_fp16.cpp"}};
            })
            .Case<IERT::TanhOp>([&](IERT::TanhOp) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"tanh_fp16"}, {"tanh_fp16.cpp"}};
            })
            .Case<IERT::HSwishOp>([&](IERT::HSwishOp) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"hswish_fp16"}, {"hswish_fp16.cpp"}};
            })
            .Case<IERT::SigmoidOp>([&](IERT::SigmoidOp) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"sigmoid_fp16"}, {"sigmoid_fp16.c"}};
            })
            .Case<IERT::SoftMaxOp>([&](IERT::SoftMaxOp softmax) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{softmax.axisIndAttr()},
                                        {"singleShaveSoftmax"},
                                        {"single_shave_softmax.cpp"}};
            })
            .Case<IERT::EluOp>([&](IERT::EluOp elu) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{elu.xAttr()}, {"elu_fp16"}, {"elu_fp16.cpp"}};
            })
            .Default([](mlir::Operation* unknownOp) -> IERT::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}

}  // namespace VPUIP
}  // namespace vpux
