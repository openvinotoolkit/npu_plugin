//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace mlir;

namespace {
// special format of dims/order available only on kernel-FW side
int64_t computeReverseMemDim(mlir::Value tensorArg, int64_t dimIdx) {
    const auto inOrder = DimsOrder::fromValue(tensorArg);
    MemDim md = inOrder.toMemDim(Dim(dimIdx));

    const auto shape = getShape(tensorArg);
    auto nDims = checked_cast<uint32_t>(shape.size());
    return nDims - 1 - md.ind();
}

static SmallVector<int64_t> reversePermutation(mlir::AffineMap map) {
    const auto origPerm = DimsOrder::fromAffineMap(map).toPermutation();
    SmallVector<int64_t> revPerm(origPerm.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = origPerm.size() - 1 - dstInd;
        revPerm[revSrcInd] = revDstInd;
    }

    return revPerm;
}

// permute int array attribute in the physical order
static SmallVector<int64_t> permuteIntArrayAttr(DimsOrder inOrder, mlir::ArrayAttr arrayAttr) {
    const auto origPerm = inOrder.toPermutation();
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> permArray(arrayAttr.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revSrcInd] = origArray[revDstInd];
    }
    return permArray;
}

}  // namespace

namespace vpux {
namespace VPUIP {

VPUIP::BlobWriter::SpecificTask SwKernelOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    return writer.createSW_KernelTask(*this);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex) {
    mlir::Value profiling_output = nullptr;

    build(builder, opState, inputs, results, profiling_output, kernelFunction, tileIndex);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::Value profiling_output, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex) {
    build(builder, opState, results.getTypes(), (profiling_output ? profiling_output.getType() : nullptr),
          kernelFunction, inputs, results, profiling_output, tileIndex);

    opState.addAttribute(result_segment_sizesAttrName(opState.name),
                         builder.getI32VectorAttr({static_cast<int32_t>(results.size()), (profiling_output ? 1 : 0)}));
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

    if (swKernelOp.profiling_data() != nullptr) {
        inferredTypes.push_back(swKernelOp.profiling_data().getType());
    }

    return mlir::success();
}

VPUIP::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    mlir::MLIRContext* ctx = origOp->getContext();

    return llvm::TypeSwitch<mlir::Operation*, VPUIP::KernelInfo>(origOp)
            .Case<VPU::ExpOp>([&](VPU::ExpOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"exp_fp16"}, {"exp_fp16.cpp"}};
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp gather) {
                const auto axisParam = computeReverseMemDim(gather.input(), gather.axis_value().getValue());
                const auto axisParamAttr = getIntAttr(gather.getContext(), axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, gather.batch_dimsAttr()},
                                         {"single_shave_gather"},
                                         {"single_shave_gather.cpp"}};
            })
            .Case<VPU::TanhOp>([&](VPU::TanhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"tanh_fp16"}, {"tanh_fp16.cpp"}};
            })
            .Case<VPU::HSwishOp>([&](VPU::HSwishOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"hswish_fp16"}, {"hswish_fp16.cpp"}};
            })
            .Case<VPU::SigmoidOp>([&](VPU::SigmoidOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sigmoid_fp16"}, {"sigmoid_fp16.c"}};
            })
            .Case<VPU::HardSigmoidOp>([&](VPU::HardSigmoidOp hardsigmoid) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{hardsigmoid.alpha_valueAttr(), hardsigmoid.beta_valueAttr()},
                        {"hardsigmoid_fp16"},
                        {"hardsigmoid_fp16.cpp"}};
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp softmax) {
                // input tensor, to transform axis
                const auto axisParam = computeReverseMemDim(softmax.input(), softmax.axisInd());
                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"singleShaveSoftmax"},
                                         {"single_shave_softmax.cpp"}};
            })
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolate) {
                const auto mode = static_cast<int64_t>(interpolate.attr().mode().getValue());
                const auto coordMode = static_cast<int64_t>(interpolate.attr().coord_mode().getValue());
                const auto nearestMode = static_cast<int64_t>(interpolate.attr().nearest_mode().getValue());
                const auto antialias = static_cast<int64_t>(interpolate.attr().antialias().getValue());

                const auto modeAttr = getIntAttr(ctx, mode);
                const auto coordModeAttr = getIntAttr(ctx, coordMode);
                const auto nearestModeAttr = getIntAttr(ctx, nearestMode);
                const auto antialiasAttr = getIntAttr(ctx, antialias);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{modeAttr, coordModeAttr, nearestModeAttr, antialiasAttr},
                        {"singleShaveInterpolate"},
                        {"single_shave_interpolate.cpp"}};
            })
            .Case<VPU::ScatterNDUpdateOp>([&](VPU::ScatterNDUpdateOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_scatterNDUpdate"},
                                         {"single_shave_scatterNDUpdate.cpp"}};
            })
            .Case<VPU::EluOp>([&](VPU::EluOp elu) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{elu.xAttr()}, {"elu_fp16"}, {"elu_fp16.cpp"}};
            })
            .Case<VPU::SqrtOp>([&](VPU::SqrtOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sqrt_fp16"}, {"sqrt_fp16.cpp"}};
            })
            .Case<VPU::DivideOp>([&](VPU::DivideOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_div_fp16"},
                                         {"eltwise_div_fp16.cpp"}};
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_mul_fp16"},
                                         {"eltwise_mul_fp16.cpp"}};
            })
            .Case<VPU::AddOp>([&](VPU::AddOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_add_fp16"},
                                         {"eltwise_add_fp16.cpp"}};
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_sub_fp16"},
                                         {"eltwise_sub_fp16.cpp"}};
            })
            .Case<VPU::MinimumOp>([&](VPU::MinimumOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_min_fp16"},
                                         {"eltwise_min_fp16.cpp"}};
            })
            .Case<VPU::MaximumOp>([&](VPU::MaximumOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_max_fp16"},
                                         {"eltwise_max_fp16.cpp"}};
            })
            .Case<VPU::PowerOp>([&](VPU::PowerOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_power_fp16"},
                                         {"eltwise_power_fp16.cpp"}};
            })
            .Case<VPU::MishOp>([&](VPU::MishOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"mish_fp16"}, {"mish_fp16.cpp"}};
            })
            .Case<VPU::MVNOp>([&](VPU::MVNOp mvn) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mvn.across_channelsAttr(),
                                                                      mvn.normalize_varianceAttr(), mvn.epsAttr()},
                                         {"singleShaveMVN"},
                                         {"single_shave_MVN.cpp"}};
            })
            .Case<VPU::MemPermuteOp>([&](VPU::MemPermuteOp permute) {
                auto memPermArr = reversePermutation(permute.mem_perm());
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntArrayAttr(ctx, memPermArr)},
                                         {"reorder_fp16"},
                                         {"reorder_fp16.cpp"}};
            })
            .Case<VPU::LRNOp>([&](VPU::LRNOp LRN) {
                const auto iType = LRN.input().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getRank() == 4 || iType.getRank() == 3,
                                  "Supporting only 4D and 3D input, got {0}", iType.getRank());
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{LRN.alphaAttr(), LRN.betaAttr(), LRN.biasAttr(), LRN.sizeAttr()},
                        {"single_shave_LRN"},
                        {"single_shave_LRN.cpp"}};
            })
            .Case<VPU::ConvertOp>([&](VPU::ConvertOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_convert"},
                                         {"single_shave_convert.cpp"}};
            })
            .Case<VPU::TopKOp>([&](VPU::TopKOp topk) {
                const auto axisParam = computeReverseMemDim(topk.input(), topk.axis());
                const auto mode = static_cast<int64_t>(topk.modeAttr().getValue());
                const auto sortMode = static_cast<int64_t>(topk.sortAttr().getValue());

                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const auto modeIntAttr = getIntAttr(ctx, mode);
                const auto sortModeAttr = getIntAttr(ctx, sortMode);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, modeIntAttr, sortModeAttr},
                                         {"single_shave_topk"},
                                         {"single_shave_topk.cpp"}};
            })
            .Case<VPU::PadOp>([&](VPU::PadOp pad) {
                mlir::MLIRContext* ctx = origOp->getContext();
                const auto inOrder = DimsOrder::fromValue(pad.input());
                const auto padBegin = permuteIntArrayAttr(inOrder, pad.pads_begin_attr().getValue());
                const auto padEnd = permuteIntArrayAttr(inOrder, pad.pads_end_attr().getValue());
                const auto padMode = static_cast<int64_t>(pad.modeAttr().getValue());

                const auto padBeginAttr = getIntArrayAttr(ctx, padBegin);
                const auto padEndAttr = getIntArrayAttr(ctx, padEnd);
                const auto padModeAttr = getIntAttr(ctx, padMode);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{padBeginAttr, padEndAttr, pad.pad_value_attrAttr(), padModeAttr},
                        {"single_shave_pad"},
                        {"single_shave_pad.cpp"}};
            })
            .Case<VPU::AvgPoolOp>([&](VPU::AvgPoolOp op) {
                auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.stridesAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_beginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_endAttr());
                const auto excludePads = static_cast<int64_t>(op.exclude_padsAttr() != nullptr);

                const auto iType = op.input().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                auto isZero = [](auto val) {
                    return val == 0;
                };
                VPUX_THROW_UNLESS(llvm::all_of(padsBegin, isZero) && llvm::all_of(padsEnd, isZero),
                                  "Padding not supported yet");

                const auto kernelSizeAttr = getIntArrayAttr(ctx, kernelSize);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto excludePadsAttr = getIntAttr(ctx, excludePads);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, padsBeginAttr,
                                                                      padsEndAttr, excludePadsAttr},
                                         {"single_shave_avg_pool"},
                                         {"single_shave_avg_pool.cpp"}};
            })
            .Case<VPU::SeluOp>([&](VPU::SeluOp selu) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{selu.alpha_valueAttr(), selu.lambda_valueAttr()},
                                         {"selu_fp16"},
                                         {"selu_fp16.cpp"}};
            })
            .Case<VPU::ReLUOp>([&](VPU::ReLUOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"relu_fp16"}, {"relu_fp16.cpp"}};
            })
            .Default([](mlir::Operation* unknownOp) -> VPUIP::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}

}  // namespace VPUIP
}  // namespace vpux