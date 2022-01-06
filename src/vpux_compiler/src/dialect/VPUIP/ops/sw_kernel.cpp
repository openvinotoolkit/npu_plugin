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

    for (auto out : swKernelOp.output_buffs()) {
        inferredTypes.push_back(out.getType());
    }

    return mlir::success();
}

IERT::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    return llvm::TypeSwitch<mlir::Operation*, IERT::KernelInfo>(origOp)
            .Case<IERT::TopKOp>([&](IERT::TopKOp topk) {
                return IERT::KernelInfo{SmallVector<mlir::Attribute>{topk.axisAttr(), topk.modeAttr(), topk.sortAttr(),
                                                                     topk.element_typeAttr()},
                                        {"single_shave_topk"},
                                        {"single_shave_topk.cpp"}};
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
            .Default([](mlir::Operation* unknownOp) -> IERT::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}

}  // namespace VPUIP
}  // namespace vpux
