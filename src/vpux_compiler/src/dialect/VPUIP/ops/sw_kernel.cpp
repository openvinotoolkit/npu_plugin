//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include <llvm/ADT/TypeSwitch.h>
#define REGION_YOLO_MAX_MASK_SIZE 9     // max mask size for region yolo op
#define REGION_YOLO_MAX_ANCHOR_SIZE 18  // max anchor size for region yolo op

using namespace vpux;
using namespace mlir;

namespace {
// special format of dims/order available only on kernel-FW side
int64_t computeReverseMemDim(mlir::Value tensorArg, int64_t dimIdx) {
    const auto inOrder = DimsOrder::fromValue(tensorArg);
    // Negative value means counting dimension from the end
    if (dimIdx < 0) {
        dimIdx += inOrder.numDims();
    }
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

// pad the int array attribute with 0 to 4 to match StridedSlice begins ends and strides param
static SmallVector<int64_t> padIntArrayAttr(mlir::ArrayAttr arrayAttr) {
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> padArray(4);
    for (const auto srcInd : irange(arrayAttr.size())) {
        padArray[srcInd] = origArray[srcInd];
    }
    return padArray;
}

static SmallVector<int64_t> getAxesArrayRevertAndOrderAware(mlir::Value tensorArg, mlir::ArrayAttr arrayAttr) {
    const auto axes = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> revertedAxesArray(MAX_NUM_DIMS, 0);
    for (const auto srcInd : irange(arrayAttr.size())) {
        revertedAxesArray[srcInd] = computeReverseMemDim(tensorArg, axes[srcInd]);
    }
    return revertedAxesArray;
}

template <class T>
void packAsFP16IntoU64(const SmallVector<T>& values, SmallVector<int64_t>& params) {
    static constexpr uint32_t PACKED_VALUES_COUNT = 4;
    float16 f16Value[PACKED_VALUES_COUNT];
    size_t packIdx = 0;

    auto f16Pack = [](float16 f16Vals[PACKED_VALUES_COUNT]) -> uint64_t {
        uint64_t ret = (static_cast<uint64_t>(f16Vals[0].to_bits()) << 0LL) |
                       (static_cast<uint64_t>(f16Vals[1].to_bits()) << 16LL) |
                       (static_cast<uint64_t>(f16Vals[2].to_bits()) << 32LL) |
                       (static_cast<uint64_t>(f16Vals[3].to_bits()) << 48LL);
        return ret;
    };

    for (const auto val : values) {
        f16Value[packIdx++] = static_cast<float16>(val);
        if (packIdx == PACKED_VALUES_COUNT) {
            params.push_back(f16Pack(f16Value));
            packIdx = 0;  // reset pack index
        }
    }

    // Pad with trailing zeros up to U64 alignment
    if (packIdx) {
        while (packIdx < PACKED_VALUES_COUNT) {
            f16Value[packIdx++] = 0;
        }
        params.push_back(f16Pack(f16Value));
    }
}

void getQuantParamsAttr(mlir::MLIRContext* ctx, mlir::Type qType, mlir::ArrayAttr& paramsAttr) {
    SmallVector<double> scales;
    SmallVector<int64_t> zeroes;

    if (qType.isa<mlir::quant::UniformQuantizedType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedType>();
        scales = {quantParams.getScale()};
        zeroes = {quantParams.getZeroPoint()};
    } else if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedPerAxisType>();
        scales = {quantParams.getScales().begin(), quantParams.getScales().end()};
        zeroes = {quantParams.getZeroPoints().begin(), quantParams.getZeroPoints().end()};
    } else {
        VPUX_THROW("Unsupported quantized type {0}", qType);
    }

    // Convert & pack 4xfp16 values into one u64 word for serialization
    llvm::SmallVector<int64_t> params;
    params.push_back(scales.size());
    packAsFP16IntoU64(scales, params);
    packAsFP16IntoU64(zeroes, params);
    paramsAttr = getIntArrayAttr(ctx, params);
}

}  // namespace

namespace vpux {
namespace VPUIP {

void vpux::VPUIP::SwKernelOp::print(mlir::OpAsmPrinter& p) {
    p.printOptionalAttrDict(
            getOperation()->getAttrs(),
            /*elidedAttrs=*/{operand_segment_sizesAttrName().strref(), kernelFunctionAttrName().strref(),
                             tileIndexAttrName().strref(), stridesAttrName().strref()});
    p << ' ';
    p.printAttributeWithoutType(kernelFunctionAttr());

    auto& opBody = body();

    if (!opBody.empty()) {
        auto* entry = &opBody.front();

        unsigned opIdx = 0;
        printGroupOfOperands(p, entry, "inputs", inputs(), opIdx);
        printGroupOfOperands(p, entry, "outputs", output_buffs(), opIdx);
    }

    auto profData = profiling_data();
    if (profData) {
        p << ' ' << "profiling_data";
        p << "(";
        p << profData;
        p << ' ' << ":";
        p << ' ';
        p << ArrayRef<mlir::Type>(profData.getType());
        p << ")";
    }

    auto opStrides = strides();
    if (opStrides) {
        p << ' ' << "strides";
        p << "(";
        p << opStrides;
        p << ")";
    }

    if (tileIndex().hasValue()) {
        p << ' ' << "on";
        p << ' ' << "tile";
        p << ' ';
        p.printAttributeWithoutType(tileIndexAttr());
    }

    p.printOptionalArrowTypeList(getResultTypes());
    p.printRegion(opBody, false);
}

mlir::ParseResult vpux::VPUIP::SwKernelOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    SmallVector<mlir::OpAsmParser::Argument> blockArgs;
    SmallVector<mlir::Type> blockTypes;

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    mlir::SymbolRefAttr kernelFunctionAttr;
    if (parser.parseAttribute(kernelFunctionAttr, parser.getBuilder().getType<mlir::NoneType>(), "kernelFunction",
                              result.attributes)) {
        return mlir::failure();
    }

    int32_t inCount = 0;
    if (parseGroupOfOperands(parser, result, blockArgs, blockTypes, "inputs", inCount).failed()) {
        return mlir::failure();
    }

    int32_t outCount = 0;
    if (parseGroupOfOperands(parser, result, blockArgs, blockTypes, "outputs", outCount).failed()) {
        return mlir::failure();
    }

    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> profiling_dataOperands;
    mlir::SmallVector<mlir::Type, 1> profiling_dataTypes;
    mlir::OptionalParseResult parseResult;

    if (succeeded(parser.parseOptionalKeyword("profiling_data"))) {
        if (parser.parseLParen()) {
            return mlir::failure();
        }

        mlir::OpAsmParser::UnresolvedOperand operand;
        parseResult = parser.parseOptionalOperand(operand);
        if (parseResult.hasValue()) {
            if (failed(*parseResult)) {
                return mlir::failure();
            }
            profiling_dataOperands.push_back(operand);
        }

        if (parser.parseColon()) {
            return mlir::failure();
        }

        mlir::Type optionalType;
        parseResult = parser.parseOptionalType(optionalType);
        if (parseResult.hasValue()) {
            if (failed(*parseResult)) {
                return mlir::failure();
            }
            profiling_dataTypes.push_back(optionalType);
        }

        if (parser.parseRParen()) {
            return mlir::failure();
        }

        auto profiling_dataOperandsLoc = parser.getCurrentLocation();
        if (parser.resolveOperands(profiling_dataOperands, profiling_dataTypes, profiling_dataOperandsLoc,
                                   result.operands)) {
            return mlir::failure();
        }
    }

    if (succeeded(parser.parseOptionalKeyword("strides"))) {
        if (parser.parseLParen()) {
            return mlir::failure();
        }

        mlir::ArrayAttr stridesAttr;
        parseResult = parser.parseOptionalAttribute(stridesAttr);
        if (!parseResult.hasValue() || failed(*parseResult)) {
            return mlir::failure();
        }

        if (parser.parseRParen()) {
            return mlir::failure();
        }
        result.attributes.set("strides", stridesAttr);
    }

    if (succeeded(parser.parseOptionalKeyword("on"))) {
        if (parser.parseKeyword("tile")) {
            return mlir::failure();
        }

        mlir::IntegerAttr tileIndexAttr;
        parseResult = parser.parseOptionalAttribute(tileIndexAttr, parser.getBuilder().getIntegerType(64), "tileIndex",
                                                    result.attributes);
        if (parseResult.hasValue() && failed(*parseResult)) {
            return mlir::failure();
        }
    }

    // Add derived `operand_segment_sizes` attribute based on parsed operands.
    auto operandSegmentSizes =
            mlir::DenseIntElementsAttr::get(mlir::VectorType::get({3}, parser.getBuilder().getI32Type()),
                                            {inCount, outCount, static_cast<int32_t>(profiling_dataOperands.size())});
    result.addAttribute("operand_segment_sizes", operandSegmentSizes);

    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs)) {
        return mlir::failure();
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask SwKernelOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    return writer.createSW_KernelTask(*this);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex) {
    mlir::Value profiling_output = nullptr;

    build(builder, opState, inputs, results, profiling_output, kernelFunction, tileIndex);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex,
                       mlir::ArrayAttr stride) {
    mlir::Value profiling_output = nullptr;
    build(builder, opState, results.getTypes(), nullptr, kernelFunction, inputs, results, profiling_output, tileIndex,
          stride);
    opState.addAttribute(result_segment_sizesAttrName(opState.name),
                         builder.getI32VectorAttr({static_cast<int32_t>(results.size()), (profiling_output ? 1 : 0)}));
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::Value profiling_output, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex) {
    build(builder, opState, results.getTypes(), (profiling_output ? profiling_output.getType() : nullptr),
          kernelFunction, inputs, results, profiling_output, tileIndex, nullptr);

    opState.addAttribute(result_segment_sizesAttrName(opState.name),
                         builder.getI32VectorAttr({static_cast<int32_t>(results.size()), (profiling_output ? 1 : 0)}));
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::Value profiling_output, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex, mlir::ArrayAttr stride) {
    build(builder, opState, results.getTypes(), (profiling_output ? profiling_output.getType() : nullptr),
          kernelFunction, inputs, results, profiling_output, tileIndex, stride);

    opState.addAttribute(result_segment_sizesAttrName(opState.name),
                         builder.getI32VectorAttr({static_cast<int32_t>(results.size()), (profiling_output ? 1 : 0)}));
}

mlir::LogicalResult SwKernelOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                 mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                 mlir::RegionRange /*regions*/,
                                                 mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

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

#define CASE_REDUCE(_OP_, _STR1_, _STR2_, _CTX_)                                                  \
    .Case<_OP_>([&](_OP_ reduce) {                                                                \
        const auto keepDims = static_cast<int64_t>(reduce.keep_dimsAttr() != nullptr);            \
        const auto keepDimsAttr = getIntAttr(_CTX_, keepDims);                                    \
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{keepDimsAttr}, {_STR1_}, {_STR2_}}; \
    })

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
            .Case<VPU::GatherElementsOp>([&](VPU::GatherElementsOp gatherElements) {
                const auto axisParam = computeReverseMemDim(gatherElements.input(), gatherElements.axis());
                const auto axisParamAttr = getIntAttr(gatherElements.getContext(), axisParam);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"single_shave_gather_elements"},
                                         {"single_shave_gather_elements.cpp"}};
            })
            .Case<VPU::GatherNDOp>([&](VPU::GatherNDOp gatherND) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{gatherND.batch_dimsAttr()},
                                         {"single_shave_gatherND"},
                                         {"single_shave_gatherND.cpp"}};
            })
            .Case<VPU::GatherTreeOp>([&](VPU::GatherTreeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_gather_tree"},
                                         {"single_shave_gather_tree.cpp"}};
            })
            .Case<VPU::ScatterUpdateOp>([&](VPU::ScatterUpdateOp scatterUpdate) {
                const auto axisParam =
                        computeReverseMemDim(scatterUpdate.input(), scatterUpdate.axis_value().getValue());
                const auto axisParamAttr = getIntAttr(scatterUpdate.getContext(), axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"single_shave_scatter_update"},
                                         {"single_shave_scatter_update.cpp"}};
            })
            .Case<VPU::TanhOp>([&](VPU::TanhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"tanh_fp16"}, {"tanh_fp16.cpp"}};
            })
            .Case<VPU::HSwishOp>([&](VPU::HSwishOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"hswish_fp16"}, {"hswish_fp16.cpp"}};
            })
            .Case<VPU::HSigmoidOp>([&](VPU::HSigmoidOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"hsigmoid_fp16"}, {"hsigmoid_fp16.cpp"}};
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
            .Case<VPU::GridSampleOp>([&](VPU::GridSampleOp gridSample) {
                const auto alignCorners = static_cast<int64_t>(gridSample.align_cornersAttr() != nullptr);
                const auto alignCornersAttr = getIntAttr(ctx, alignCorners);

                int64_t mode = 0;
                if (gridSample.modeAttr() != nullptr) {
                    mode = static_cast<int64_t>(gridSample.modeAttr().getValue());
                }
                const auto modeIntAttr = getIntAttr(ctx, mode);

                int64_t paddingMode = 0;
                if (gridSample.padding_modeAttr() != nullptr) {
                    paddingMode = static_cast<int64_t>(gridSample.padding_modeAttr().getValue());
                }
                const auto paddingModeAttr = getIntAttr(ctx, paddingMode);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{alignCornersAttr, modeIntAttr, paddingModeAttr},
                                         {"single_shave_grid_sample"},
                                         {"single_shave_grid_sample.cpp"}};
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp softmax) {
                // input tensor, to transform axis
                const auto axisParam = computeReverseMemDim(softmax.input(), softmax.axisInd());
                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"singleShaveSoftmax"},
                                         {"single_shave_softmax.cpp"}};
            })
            .Case<VPU::LogSoftmaxOp>([&](VPU::LogSoftmaxOp logSoftmax) {
                const auto axisParam = computeReverseMemDim(logSoftmax.input(), logSoftmax.axisInd());
                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"singleShaveLogSoftmax"},
                                         {"single_shave_log_softmax.cpp"}};
            })
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolate) {
                const auto mode = static_cast<int64_t>(interpolate.attr().getMode().getValue());
                const auto coordMode = static_cast<int64_t>(interpolate.attr().getCoordMode().getValue());
                const auto nearestMode = static_cast<int64_t>(interpolate.attr().getNearestMode().getValue());
                const auto antialias = static_cast<int64_t>(interpolate.attr().getAntialias().getValue());
                const auto inOrder = DimsOrder::fromValue(interpolate.input());
                const auto initialInputDim = interpolate.initial_input_dims_attr().getValue();
                const auto initialInputDimsParam = permuteIntArrayAttr(inOrder, initialInputDim);
                const auto initialOutputDim = interpolate.initial_output_dims_attr().getValue();
                const auto initialOutputDimsParam = permuteIntArrayAttr(inOrder, initialOutputDim);
                const auto tileFPList = parseFPArrayAttr<double>(interpolate.tile_offset_attrAttr());
                const auto cubeCoeffParam = interpolate.attr().getCubeCoeff().getValueAsDouble();
                const auto axisParam = parseIntArrayAttr<int64_t>(interpolate.axes_attrAttr());
                // Check the scaling dim size since the swkernel only support scaling at most two dims
                SmallVector<int64_t> scalingAxis;
                for (auto axis : axisParam) {
                    if (initialInputDim[axis] != initialOutputDim[axis]) {
                        scalingAxis.push_back(axis);
                    }
                }
                VPUX_THROW_WHEN(scalingAxis.size() > 2, "Unsupported scaling dim size {0} at '{1}'", scalingAxis.size(),
                                interpolate->getLoc());

                const auto initialInputOffset =
                        interpolate.initial_input_offset_attr().hasValue()
                                ? permuteIntArrayAttr(inOrder, interpolate.initial_input_offset_attr().getValue())
                                : SmallVector<int64_t>(inOrder.numDims(), 0);
                const auto initialOutputOffset =
                        interpolate.initial_output_offset_attr().hasValue()
                                ? permuteIntArrayAttr(inOrder, interpolate.initial_output_offset_attr().getValue())
                                : SmallVector<int64_t>(inOrder.numDims(), 0);

                const auto modeAttr = getIntAttr(ctx, mode);
                const auto coordModeAttr = getIntAttr(ctx, coordMode);
                const auto nearestModeAttr = getIntAttr(ctx, nearestMode);
                const auto antialiasAttr = getIntAttr(ctx, antialias);
                const auto tileAttr = getFPArrayAttr(ctx, tileFPList);
                const auto initialInputDimsParamAttr = getIntArrayAttr(ctx, initialInputDimsParam);
                const auto initialOutputDimsParamAttr = getIntArrayAttr(ctx, initialOutputDimsParam);
                const auto cubeCoeffParamAttr = getFPAttr(ctx, cubeCoeffParam);
                const auto axisParamAttr = getIntArrayAttr(ctx, scalingAxis);
                const auto initialInputOffsetAttr = getIntArrayAttr(ctx, initialInputOffset);
                const auto initialOutputOffsetAttr = getIntArrayAttr(ctx, initialOutputOffset);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{
                                                 modeAttr, coordModeAttr, nearestModeAttr, antialiasAttr, tileAttr,
                                                 initialInputDimsParamAttr, initialOutputDimsParamAttr, axisParamAttr,
                                                 cubeCoeffParamAttr, initialInputOffsetAttr, initialOutputOffsetAttr},
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
            .Case<VPU::ClampOp>([&](VPU::ClampOp clamp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{clamp.minAttr(), clamp.maxAttr()},
                                         {"clamp_fp16"},
                                         {"clamp_fp16.cpp"}};
            })
            .Case<VPU::FullyConnectedOp>([&](VPU::FullyConnectedOp fullyConnected) {
                const auto inputs = fullyConnected.getInputs();
                const auto inOrder = DimsOrder::fromValue(fullyConnected.input());

                VPUX_THROW_UNLESS(inOrder == DimsOrder::NC, "Layout not supported, got {0}, expected NC", inOrder);

                // Serialization of optional arguments for sw operators not supported
                // [E-61263]
                VPUX_THROW_UNLESS(inputs.size() == 2,
                                  "Optional bias case not supported, got {0} number of inputs, expected 2",
                                  inputs.size());
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"fully_connected"}, {"fully_connected.cpp"}};
            })
            .Case<VPU::SqrtOp>([&](VPU::SqrtOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sqrt_fp16"}, {"sqrt_fp16.cpp"}};
            })
            .Case<VPU::ErfOp>([&](VPU::ErfOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"erf_fp16"}, {"erf_fp16.cpp"}};
            })
            .Case<VPU::CeilingOp>([&](VPU::CeilingOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"ceil_fp16"}, {"ceil_fp16.cpp"}};
            })
            .Case<VPU::DivideOp>([&](VPU::DivideOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_div"}, {"eltwise_div.cpp"}};
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_mul"}, {"eltwise_mul.cpp"}};
            })
            .Case<VPU::AddOp>([&](VPU::AddOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_add"}, {"eltwise_add.cpp"}};
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_sub"}, {"eltwise_sub.cpp"}};
            })
            .Case<VPU::MinimumOp>([&](VPU::MinimumOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_min"}, {"eltwise_min.cpp"}};
            })
            .Case<VPU::MaximumOp>([&](VPU::MaximumOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_max"}, {"eltwise_max.cpp"}};
            })
            .Case<VPU::PowerOp>([&](VPU::PowerOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_power"}, {"eltwise_power.cpp"}};
            })
            .Case<VPU::EqualOp>([&](VPU::EqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_equal"}, {"eltwise_equal.cpp"}};
            })
            .Case<VPU::FloorModOp>([&](VPU::FloorModOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_floor_mod"},
                                         {"eltwise_floor_mod_.cpp"}};
            })
            .Case<VPU::GreaterOp>([&](VPU::GreaterOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_greater"}, {"eltwise_greater.cpp"}};
            })
            .Case<VPU::GreaterEqualOp>([&](VPU::GreaterEqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_greater_equal"},
                                         {"eltwise_greater_equal.cpp"}};
            })
            .Case<VPU::LessOp>([&](VPU::LessOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_less"}, {"eltwise_less.cpp"}};
            })
            .Case<VPU::LessEqualOp>([&](VPU::LessEqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_less_equal"},
                                         {"eltwise_less_equal.cpp"}};
            })
            .Case<VPU::LogicalOrOp>([&](VPU::LogicalOrOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_logical_or"},
                                         {"eltwise_logical_or.cpp"}};
            })
            .Case<VPU::LogicalXorOp>([&](VPU::LogicalXorOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_logical_xor"},
                                         {"eltwise_logical_xor.cpp"}};
            })
            .Case<VPU::LogicalNotOp>([&](VPU::LogicalNotOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_logical_not"},
                                         {"eltwise_logical_not.cpp"}};
            })
            .Case<VPU::AndOp>([&](VPU::AndOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_and"}, {"eltwise_and.cpp"}};
            })
            .Case<VPU::NotEqualOp>([&](VPU::NotEqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_not_equal"},
                                         {"eltwise_not_equal.cpp"}};
            })
            .Case<VPU::MishOp>([&](VPU::MishOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"mish_fp16"}, {"mish_fp16.cpp"}};
            })
            .Case<VPU::MVNOp>([&](VPU::MVNOp mvn) {
                const auto iType = mvn.input().getType().cast<vpux::NDTypeInterface>();
                const auto oType = mvn.output().getType().cast<vpux::NDTypeInterface>();
                const auto iOrder = iType.getDimsOrder();
                const auto supported = {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC};
                VPUX_THROW_UNLESS(llvm::any_of(supported,
                                               [iOrder](DimsOrder order) {
                                                   return order == iOrder;
                                               }),
                                  "Unsupported order {0}", iOrder);

                if (iOrder == DimsOrder::NCHW || iOrder == DimsOrder::NCWH) {
                    const auto compact = StrideReqs::compact(iType.getRank());
                    VPUX_THROW_UNLESS(compact.checkStrides(iType), "Only compact input supported, got {0}", iType);
                    VPUX_THROW_UNLESS(compact.checkStrides(oType), "Only compact output supported, got {0}", oType);
                }

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mvn.across_channelsAttr(),
                                                                      mvn.normalize_varianceAttr(), mvn.epsAttr()},
                                         {"singleShaveMVN"},
                                         {"singleShaveMVN.cpp"}};
            })
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op mvn) {
                mlir::MLIRContext* ctx = mvn.getContext();
                // Convert 'axes' to physical reversed (innermost first) equivalent
                const auto axesVal = parseIntArrayAttr<int64_t>(mvn.axes());
                SmallVector<int64_t> memAxes;
                for (const auto axis : axesVal) {
                    memAxes.push_back(computeReverseMemDim(mvn.input(), axis));
                }
                const auto numAxes = memAxes.size();
                const auto epsMode = static_cast<int64_t>(mvn.eps_modeAttr().getValue());
                const auto memAxesAttr = getIntArrayAttr(ctx, memAxes);
                const auto numAxesAttr = getIntAttr(ctx, numAxes);
                const auto epsModeAttr = getIntAttr(ctx, epsMode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mvn.normalize_varianceAttr(), epsModeAttr,
                                                                      mvn.epsAttr(), numAxesAttr, memAxesAttr},
                                         {"mvn6_fp16"},
                                         {"mvn6_fp16.cpp"}};
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
            .Case<VPU::RegionYoloOp>([&](VPU::RegionYoloOp regionYolo) {
                // Add maskSize as input params
                const auto maskSize = static_cast<int64_t>(regionYolo.maskAttr().getValue().size());
                const auto maskSizeParamAttr = getIntAttr(ctx, maskSize);
                // Supplement for maskAttrInput to meet array size(9) defined in kernel
                const auto maskVec = parseIntArrayAttr<int64_t>(regionYolo.maskAttr());
                assert(maskSize <= REGION_YOLO_MAX_MASK_SIZE);
                int64_t maskList[REGION_YOLO_MAX_MASK_SIZE];
                for (int i = 0; i < REGION_YOLO_MAX_MASK_SIZE; i++) {
                    if (i >= maskSize) {
                        maskList[i] = 0;
                    } else {
                        maskList[i] = maskVec[i];
                    }
                }
                const auto maskArrayRef = ArrayRef<int64_t>(maskList, REGION_YOLO_MAX_MASK_SIZE);
                const auto maskAttrInput = getIntArrayAttr(ctx, maskArrayRef);
                // Supplement for anchorAttrInput to meet array size(18) defined in kernel
                const auto anchorSize = static_cast<int64_t>(regionYolo.anchorsAttr().getValue().size());
                assert(anchorSize <= REGION_YOLO_MAX_ANCHOR_SIZE);
                float anchorList[REGION_YOLO_MAX_ANCHOR_SIZE];
                const auto anchorFloatList = parseFPArrayAttr<float>(regionYolo.anchorsAttr());
                for (int i = 0; i < REGION_YOLO_MAX_ANCHOR_SIZE; i++) {
                    if (i >= anchorSize) {
                        anchorList[i] = 0;
                    } else {
                        anchorList[i] = anchorFloatList[i];
                    }
                }
                const auto anchorArrayRef = ArrayRef<float>(anchorList, REGION_YOLO_MAX_ANCHOR_SIZE);
                const auto anchorAttrInput = getIntArrayAttr(ctx, anchorArrayRef);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{regionYolo.coordsAttr(), regionYolo.classesAttr(),
                                                     regionYolo.regionsAttr(), regionYolo.do_softmaxAttr(),
                                                     maskSizeParamAttr, maskAttrInput, regionYolo.axisAttr(),
                                                     regionYolo.end_axisAttr(), anchorAttrInput},
                        {"single_shave_region_yolo"},
                        {"single_shave_region_yolo.cpp"}};
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
            .Case<VPU::ExtractImagePatchesOp>([&](VPU::ExtractImagePatchesOp op) {
                auto sizes = parseIntArrayAttr<int64_t>(op.sizesAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.stridesAttr());
                auto rates = parseIntArrayAttr<int64_t>(op.ratesAttr());
                const auto autoPad = static_cast<int32_t>(op.autoPadAttr().getValue());

                const auto iType = op.data().getType().cast<vpux::NDTypeInterface>();
                const auto iOrder = iType.getDimsOrder();
                const auto supported = {DimsOrder::NCHW};
                VPUX_THROW_UNLESS(llvm::any_of(supported,
                                               [iOrder](DimsOrder order) {
                                                   return order == iOrder;
                                               }),
                                  "Unsupported order {0}", iOrder);

                const auto kernelSizeAttr = getIntArrayAttr(ctx, sizes);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto ratesAttr = getIntArrayAttr(ctx, rates);
                const auto autoPadAttr = getIntAttr(ctx, autoPad);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, ratesAttr, autoPadAttr},
                        {"single_shave_extract_image_patches"},
                        {"single_shave_extract_image_patches.cpp"}};
            })
            .Case<VPU::PReluOp>([&](VPU::PReluOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"prelu_fp16"}, {"prelu_fp16.cpp"}};
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
            .Case<VPU::FakeQuantizeOp>([&](VPU::FakeQuantizeOp op) {
                const auto iType = op.input().getType().cast<vpux::NDTypeInterface>();
                const auto oType = op.output().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getElementType().isF16() && oType.getElementType().isF16(),
                                  "Only supports FP16 in/out");
                const auto levelsAttr = getIntAttr(ctx, op.levels());
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{levelsAttr},
                                         {"fake_quantize"},
                                         {"fake_quantize.cpp"}};
            })
            .Case<VPU::QuantizeOp>([&](VPU::QuantizeOp op) {
                const auto oType = op.output().getType().cast<vpux::NDTypeInterface>();
                mlir::ArrayAttr paramsAttr;
                getQuantParamsAttr(ctx, oType.getElementType(), paramsAttr);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{paramsAttr}, {"quantize"}, {"quantize.cpp"}};
            })
            .Case<VPU::DequantizeOp>([&](VPU::DequantizeOp op) {
                const auto iType = op.input().getType().cast<vpux::NDTypeInterface>();
                mlir::ArrayAttr paramsAttr;
                getQuantParamsAttr(ctx, iType.getElementType(), paramsAttr);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{paramsAttr}, {"dequantize"}, {"dequantize.cpp"}};
            })
            .Case<VPU::DynamicQuantizeOp>([&](VPU::DynamicQuantizeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"dynamic_quantize"},
                                         {"dynamic_quantize.cpp"}};
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp depth_to_space) {
                const auto mode = static_cast<int64_t>(depth_to_space.modeAttr().getValue());
                const auto modeAttr = getIntAttr(ctx, mode);
                SmallVector<mlir::Attribute> paramAttr = {depth_to_space.block_sizeAttr(), modeAttr};
                if (depth_to_space.padded_channels().hasValue()) {
                    auto paddedIC = depth_to_space.padded_channels().getValue().input();
                    auto paddedOC = depth_to_space.padded_channels().getValue().output();

                    if (paddedIC != nullptr) {
                        paramAttr.push_back(paddedIC);
                    }

                    if (paddedOC != nullptr) {
                        paramAttr.push_back(paddedOC);
                    }
                }
                return VPUIP::KernelInfo{paramAttr,
                                         {"single_shave_depth_to_space"},
                                         {"single_shave_depth_to_space.cpp"}};
            })
            .Case<VPU::SpaceToDepthOp>([&](VPU::SpaceToDepthOp space_to_depth) {
                const auto mode = static_cast<int64_t>(space_to_depth.modeAttr().getValue());
                const auto modeAttr = getIntAttr(ctx, mode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{space_to_depth.block_sizeAttr(), modeAttr},
                                         {"single_shave_space_to_depth"},
                                         {"single_shave_space_to_depth.cpp"}};
            })
            .Case<VPU::SeluOp>([&](VPU::SeluOp selu) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{selu.alpha_valueAttr(), selu.lambda_valueAttr()},
                                         {"selu_fp16"},
                                         {"selu_fp16.cpp"}};
            })
            .Case<VPU::LeakyReluOp>([&](VPU::LeakyReluOp leakyRelu) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{leakyRelu.negative_slopeAttr()},
                                         {"leaky_relu_fp16"},
                                         {"leaky_relu_fp16.cpp"}};
            })
            .Case<VPU::SwishOp>([&](VPU::SwishOp swish) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{swish.beta_valueAttr()},
                                         {"swish_fp16"},
                                         {"swish_fp16.cpp"}};
            })
            .Case<VPU::ReLUOp>([&](VPU::ReLUOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"relu_fp16"}, {"relu_fp16.cpp"}};
            })
            .Case<VPU::NegativeOp>([&](VPU::NegativeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_negative"},
                                         {"single_shave_negative.cpp"}};
            })
            .Case<VPU::StridedSliceOp>([&](VPU::StridedSliceOp stridedslice) {
                mlir::MLIRContext* ctx = origOp->getContext();

                const auto stridedSliceBegins = padIntArrayAttr(stridedslice.begins_attr());
                const auto stridedSliceEnds = padIntArrayAttr(stridedslice.ends_attr());
                const auto stridedSliceStrides = padIntArrayAttr(stridedslice.strides_attr().getValue());

                const auto stridedSliceBeginsAttr = getIntArrayAttr(ctx, stridedSliceBegins);
                const auto stridedSliceEndsAttr = getIntArrayAttr(ctx, stridedSliceEnds);
                const auto stridedSliceStridesAttr = getIntArrayAttr(ctx, stridedSliceStrides);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{stridedSliceBeginsAttr, stridedSliceEndsAttr,
                                                                      stridedSliceStridesAttr},
                                         {"single_shave_stridedslice"},
                                         {"single_shave_stridedslice.cpp"}};
            })
            .Case<VPU::ReverseSequenceOp>([&](VPU::ReverseSequenceOp ReverseSequence) {
                const auto batchAxis = computeReverseMemDim(ReverseSequence.data(), ReverseSequence.batch_axis());
                const auto seqAxis = computeReverseMemDim(ReverseSequence.data(), ReverseSequence.seq_axis());
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{getIntAttr(ctx, batchAxis), getIntAttr(ctx, seqAxis)},
                        {"single_shave_reverse_sequence"},
                        {"single_shave_reverse_sequence.cpp"}};
            })
            .Case<VPU::YuvToRgbOp>([&](VPU::YuvToRgbOp yuvToRgb) {
                mlir::MLIRContext* ctx = origOp->getContext();
                const auto inFmt = yuvToRgb.inFmtAttr().getValue();
                const auto outFmt = static_cast<int64_t>(yuvToRgb.outFmtAttr().getValue()) - 2;
                const auto outFmtIntAttr = getIntAttr(ctx, outFmt);
                auto singlePlane = (yuvToRgb.input2() == nullptr);

                if (inFmt == vpux::IE::ColorFmt::NV12) {
                    if (singlePlane) {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr},
                                                 {"single_shave_convert_color_nv12_to_rgb_single_plane"},
                                                 {"single_shave_convert_color_nv12_to_rgb_single_plane.cpp"}};
                    } else {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr},
                                                 {"single_shave_convert_color_nv12_to_rgb"},
                                                 {"single_shave_convert_color_nv12_to_rgb.cpp"}};
                    }
                } else {
                    if (singlePlane) {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr},
                                                 {"single_shave_convert_color_i420_to_rgb_single_plane"},
                                                 {"single_shave_convert_color_i420_to_rgb_single_plane.cpp"}};
                    } else {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr},
                                                 {"single_shave_convert_color_i420_to_rgb"},
                                                 {"single_shave_convert_color_i420_to_rgb.cpp"}};
                    }
                }
            })
            .Case<VPU::RandomUniformOp>([&](VPU::RandomUniformOp rand) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{rand.global_seedAttr(), rand.op_seedAttr()},
                                         {"random_uniform"},
                                         {"random_uniform.cpp"}};
            })
            .Case<VPU::ROIPoolingOp>([&](VPU::ROIPoolingOp roi) {
                const auto iType = roi.input().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                const auto method = static_cast<int64_t>(roi.methodAttr().getValue());
                const auto methodAttr = getIntAttr(ctx, method);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{roi.output_sizeAttr(), roi.spatial_scaleAttr(), methodAttr},
                        {"single_shave_roipooling"},
                        {"single_shave_roipooling.cpp"}};
            })
            .Case<VPU::RollOp>([&](VPU::RollOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_roll"},
                                         {"single_shave_roll.cpp"}};
            })
            .Case<VPU::OneHotOp>([&](VPU::OneHotOp oneHot) {
                int64_t axis = oneHot.axis();
                const auto shape = getShape(oneHot.input());
                auto nDims = checked_cast<int64_t>(shape.size());
                const int64_t actualAxis = (axis < 0) ? -axis - 1 : nDims - axis;

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntAttr(ctx, actualAxis), oneHot.depthAttr(),
                                                                      oneHot.on_valueAttr(), oneHot.off_valueAttr()},
                                         {"single_shave_onehot"},
                                         {"single_shave_onehot.cpp"}};
            })
            .Case<VPU::ReorgYoloOp>([&](VPU::ReorgYoloOp reorgYolo) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{reorgYolo.strideAttr()},
                                         {"single_shave_reorg_yolo"},
                                         {"single_shave_reorg_yolo.cpp"}};
            })
            .Case<VPU::PSROIPoolingOp>([&](VPU::PSROIPoolingOp psroi) {
                const auto iType = psroi.input().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                const auto spatialBinsX = static_cast<int64_t>(psroi.spatial_bins_xAttr() != nullptr);
                const auto spatialBinsY = static_cast<int64_t>(psroi.spatial_bins_yAttr() != nullptr);
                const auto mode = static_cast<int64_t>(psroi.modeAttr() != nullptr);

                const auto spatialBinsXAttr = getIntAttr(ctx, spatialBinsX);
                const auto spatialBinsYAttr = getIntAttr(ctx, spatialBinsY);
                const auto modeAttr = getIntAttr(ctx, mode);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{psroi.output_dimAttr(), psroi.spatial_scaleAttr(),
                                                                      psroi.group_sizeAttr(), spatialBinsXAttr,
                                                                      spatialBinsYAttr, modeAttr},
                                         {"single_shave_ps_roipooling"},
                                         {"single_shave_ps_roipooling.cpp"}};
            })
            .Case<VPU::NonMaxSuppressionOp>([&](VPU::NonMaxSuppressionOp nms) {
                const auto boxEncoding = static_cast<int64_t>(nms.box_encodingAttr().getValue());
                const auto boxEncodingAttr = getIntAttr(ctx, boxEncoding);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{nms.max_output_boxes_per_class_valueAttr(),
                                                     nms.iou_threshold_valueAttr(), nms.score_threshold_valueAttr(),
                                                     nms.soft_nms_sigma_valueAttr(), boxEncodingAttr},
                        {"nms_fp16"},
                        {"nms_fp16.cpp"}};
            })
            .Case<VPU::CTCGreedyDecoderOp>([&](VPU::CTCGreedyDecoderOp op) {
                const auto mergeRepeated = static_cast<int64_t>(op.mergeRepeatedAttr() != nullptr);
                const auto mergeRepeatedAttr = getIntAttr(ctx, mergeRepeated);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mergeRepeatedAttr},
                                         {"single_shave_ctc_greedy_decoder"},
                                         {"single_shave_ctc_greedy_decoder.cpp"}};
            }) CASE_REDUCE(VPU::ReduceL1Op, "reduce_l1", "reduce_l1.cpp", ctx)
                    CASE_REDUCE(VPU::ReduceSumOp, "reduce_sum", "reduce_sum.cpp", ctx) CASE_REDUCE(
                            VPU::ReduceMeanOp, "reduce_mean", "reduce_mean.cpp", ctx)
                            CASE_REDUCE(VPU::ReduceLogicalAndOp, "reduce_and", "reduce_and.cpp", ctx) CASE_REDUCE(
                                    VPU::ReduceLogicalOrOp, "reduce_or", "reduce_or.cpp", ctx)
                                    CASE_REDUCE(VPU::ReduceMaxOp, "reduce_max", "reduce_max.cpp", ctx) CASE_REDUCE(
                                            VPU::ReduceMinOp, "reduce_min", "reduce_min.cpp", ctx)
                                            CASE_REDUCE(VPU::ReduceProdOp, "reduce_prod", "reduce_prod.cpp", ctx)
                                                    CASE_REDUCE(VPU::ReduceL2Op, "reduce_l2", "reduce_l2.cpp", ctx)
            .Case<VPU::SinOp>([&](VPU::SinOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sin_fp16"}, {"sin_fp16.cpp"}};
            })
            .Case<VPU::CosOp>([&](VPU::CosOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"cos_fp16"}, {"cos_fp16.cpp"}};
            })
            .Case<VPU::SinhOp>([&](VPU::SinhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sinh_fp16"}, {"sinh_fp16.cpp"}};
            })
            .Case<VPU::CoshOp>([&](VPU::CoshOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"cosh_fp16"}, {"cosh_fp16.cpp"}};
            })
            .Case<VPU::FloorOp>([&](VPU::FloorOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"floor_fp16"}, {"floor_fp16.cpp"}};
            })
            .Case<VPU::SignOp>([&](VPU::SignOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sign_fp16"}, {"sign_fp16.cpp"}};
            })
            .Case<VPU::TileOp>([&](VPU::TileOp tileOp) {
                auto repeats_size = static_cast<int64_t>(tileOp.repeats_valuesAttr().getValue().size());
                auto repeats_sizeAttr = getIntAttr(ctx, repeats_size);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{repeats_sizeAttr, tileOp.repeats_valuesAttr()},
                                         {"single_shave_tile"},
                                         {"single_shave_tile.cpp"}};
            })
            .Case<VPU::TanOp>([&](VPU::TanOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"tan_fp16"}, {"tan_fp16.cpp"}};
            })
            .Case<VPU::AsinOp>([&](VPU::AsinOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"asin_fp16"}, {"asin_fp16.cpp"}};
            })
            .Case<VPU::AcosOp>([&](VPU::AcosOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"acos_fp16"}, {"acos_fp16.cpp"}};
            })
            .Case<VPU::AtanOp>([&](VPU::AtanOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"atan_fp16"}, {"atan_fp16.cpp"}};
            })
            .Case<VPU::AsinhOp>([&](VPU::AsinhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"asinh_fp16"}, {"asinh_fp16.cpp"}};
            })
            .Case<VPU::AcoshOp>([&](VPU::AcoshOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"acosh_fp16"}, {"acosh_fp16.cpp"}};
            })
            .Case<VPU::AtanhOp>([&](VPU::AtanhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"atanh_fp16"}, {"atanh_fp16.cpp"}};
            })
            .Case<VPU::DetectionOutputNormalizeOp>([&](VPU::DetectionOutputNormalizeOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.input_widthAttr(), op.input_heightAttr()},
                                         {"detection_output_normalize"},
                                         {"detection_output_normalize.cpp"}};
            })
            .Case<VPU::DetectionOutputDecodeBoxesOp>([&](VPU::DetectionOutputDecodeBoxesOp op) {
                const auto codeType = [&]() -> int64_t {
                    enum CodeType : int64_t { CENTER_SIZE, CORNER, CORNER_SIZE };
                    switch (op.code_type()) {
                    case IE::DetectionOutputCodeType::CENTER_SIZE:
                        return CodeType::CENTER_SIZE;
                    case IE::DetectionOutputCodeType::CORNER:
                        return CodeType::CORNER;
                    case IE::DetectionOutputCodeType::CORNER_SIZE:
                        return CodeType::CORNER_SIZE;
                    }

                    const auto codeTypeString = stringifyDetectionOutputCodeType(op.code_type());
                    VPUX_THROW("Unsupported DetectionOutput codeType: {0}", codeTypeString);
                }();

                const auto clipBeforeNms = static_cast<int64_t>(op.clip_before_nms());
                const auto clipBeforeNmsAttr = getIntAttr(ctx, clipBeforeNms);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntAttr(ctx, codeType), clipBeforeNmsAttr},
                                         {"detection_output_decode_boxes"},
                                         {"detection_output_decode_boxes.cpp"}};
            })
            .Case<VPU::DetectionOutputSortTopKOp>([&](VPU::DetectionOutputSortTopKOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.confidence_thresholdAttr(), op.top_kAttr(),
                                                                      op.background_idAttr()},
                                         {"detection_output_sort_top_k"},
                                         {"detection_output_sort_top_k.cpp"}};
            })
            .Case<VPU::DetectionOutputSelectBoxesOp>([&](VPU::DetectionOutputSelectBoxesOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"detection_output_select_boxes"},
                                         {"detection_output_select_boxes.cpp"}};
            })
            .Case<VPU::DetectionOutputNmsCaffeOp>([&](VPU::DetectionOutputNmsCaffeOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.nms_thresholdAttr()},
                                         {"detection_output_nms_caffe"},
                                         {"detection_output_nms_caffe.cpp"}};
            })
            .Case<VPU::DetectionOutputCollectResultsOp>([&](VPU::DetectionOutputCollectResultsOp op) {
                const auto clipAfterNms = static_cast<int64_t>(op.clip_after_nms());
                const auto clipAfterNmsAttr = getIntAttr(ctx, clipAfterNms);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.keep_top_kAttr(), clipAfterNmsAttr},
                                         {"detection_output_collect_results"},
                                         {"detection_output_collect_results.cpp"}};
            })
            .Case<VPU::PermuteQuantizeOp>([&](VPU::PermuteQuantizeOp op) {
                // permutation params convert to shv order
                auto memPermArr = reversePermutation(op.mem_perm());
                //  kernel implementation allow optimized speed configuration to be specify
                enum PermuteQuantizeOptMode : int64_t {
                    // fields meaning: InOrder_OutOrder_NoChanels
                    PQ_NONE = 0,
                    PQ_NCHW_NHWC_C1 = 1,
                    PQ_NCHW_NHWC_C3 = 2,
                    PQ_NCHW_NHWC_C4 = 3,
                    PQ_NCHW_NHWC_C3EXP4 = 4,
                    PQ_NCHW_NHWC_C4EXP4 = 5,
                    PQ_NCHW_NHWC_C1EXP4 = 6
                };
                int64_t optMode = PermuteQuantizeOptMode::PQ_NONE;
                const auto iType = op.input().getType().cast<vpux::NDTypeInterface>();
                const auto oType = op.output().getType().cast<vpux::NDTypeInterface>();
                const auto inOrder = DimsOrder::fromValue(op.input());
                const auto outOrder = DimsOrder::fromValue(op.output());
                if ((inOrder == DimsOrder::NCHW) && (outOrder == DimsOrder::NHWC) &&
                    (1 == iType.getShape()[Dims4D::Act::N])) {
                    if (1 == iType.getShape()[Dims4D::Act::C]) {
                        optMode = PermuteQuantizeOptMode::PQ_NCHW_NHWC_C1;
                    } else if (3 == iType.getShape()[Dims4D::Act::C]) {
                        optMode = PermuteQuantizeOptMode::PQ_NCHW_NHWC_C3;
                    } else if (4 == iType.getShape()[Dims4D::Act::C]) {
                        optMode = PermuteQuantizeOptMode::PQ_NCHW_NHWC_C4;
                    }
                }
                // Extract quantize scale and zero
                auto quantParams = oType.getElementType().cast<mlir::quant::UniformQuantizedType>();
                double scale = quantParams.getScale();
                int64_t zero = quantParams.getZeroPoint();

                // Custom kernels speed up for scenario of expand to 4.
                // Reduce kernel rump-up by enabling just higher resolutions that kernel parallelism allows.
                // Haven't implement trailing elements management. If resolution is not multiple of 16 will be managed
                // by default mode.
                auto szwh = oType.getShape()[Dims4D::Act::H] * oType.getShape()[Dims4D::Act::W];
                if ((PermuteQuantizeOptMode::PQ_NCHW_NHWC_C3 == optMode) && (4 == oType.getShape()[Dims4D::Act::C]) &&
                    (0 == (szwh % 16)) && (szwh >= 48)) {
                    optMode = PQ_NCHW_NHWC_C3EXP4;
                }
                if ((PermuteQuantizeOptMode::PQ_NCHW_NHWC_C4 == optMode) && (4 == oType.getShape()[Dims4D::Act::C]) &&
                    (0 == (szwh % 8)) && (szwh >= 48)) {
                    optMode = PQ_NCHW_NHWC_C4EXP4;
                }
                if ((PermuteQuantizeOptMode::PQ_NCHW_NHWC_C1 == optMode) && (4 == oType.getShape()[Dims4D::Act::C]) &&
                    (0 == (szwh % 16)) && (szwh >= 64)) {
                    optMode = PQ_NCHW_NHWC_C1EXP4;
                }
                if (iType.getElementType().isF32()) {
                    return VPUIP::KernelInfo{
                            SmallVector<mlir::Attribute>{getIntAttr(ctx, optMode), getFPAttr(ctx, scale),
                                                         getIntAttr(ctx, zero), getIntArrayAttr(ctx, memPermArr)},
                            {"permute_quantize_fp32"},
                            {"permute_quantize_fp32.c"}};
                } else {
                    return VPUIP::KernelInfo{
                            SmallVector<mlir::Attribute>{getIntAttr(ctx, optMode), getFPAttr(ctx, scale),
                                                         getIntAttr(ctx, zero), getIntArrayAttr(ctx, memPermArr)},
                            {"permute_quantize"},
                            {"permute_quantize.c"}};
                }
            })
            .Case<VPU::RoundOp>([&](VPU::RoundOp round) {
                const auto mode = static_cast<int64_t>(round.modeAttr().getValue());
                const auto modeIntAttr = getIntAttr(ctx, mode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{modeIntAttr}, {"round_fp16"}, {"round_fp16.cpp"}};
            })
            .Case<VPU::LogOp>([&](VPU::LogOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"log_fp16"}, {"log_fp16.cpp"}};
            })
            .Case<VPU::NormalizeL2Op>([&](VPU::NormalizeL2Op normalizeL2) {
                const auto epsMode = static_cast<int64_t>(normalizeL2.eps_modeAttr().getValue());
                const auto epsModeAttr = getIntAttr(ctx, epsMode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{normalizeL2.epsAttr(), epsModeAttr},
                                         {"normalize_l2_fp16"},
                                         {"normalize_l2_fp16.cpp"}};
            })
            .Case<VPU::CumSumOp>([&](VPU::CumSumOp cumSum) {
                const auto axisParam = computeReverseMemDim(cumSum.input(), cumSum.axis_value().getValue());
                const auto exclusive = static_cast<int64_t>(cumSum.exclusiveAttr() != nullptr);
                const auto reverse = static_cast<int64_t>(cumSum.reverseAttr() != nullptr);

                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const auto reverseAttr = getIntAttr(ctx, reverse);
                const auto exclusiveAttr = getIntAttr(ctx, exclusive);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, exclusiveAttr, reverseAttr},
                                         {"cum_sum_fp16"},
                                         {"cum_sum_fp16.cpp"}};
            })
            .Case<VPU::SelectOp>([&](VPU::SelectOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_select_fp16"},
                                         {"eltwise_select_fp16.cpp"}};
            })
            .Case<VPU::SoftPlusOp>([&](VPU::SoftPlusOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"softplus_fp16"}, {"softplus_fp16.cpp"}};
            })
            .Case<VPU::EmbeddingBagPackedSumOp>([&](VPU::EmbeddingBagPackedSumOp emb) {
                const auto inputs = emb.getInputs();
                // Serialization of optional arguments for sw operators not supported
                // [E-61263]
                VPUX_THROW_UNLESS(inputs.size() == 3,
                                  "Optional weights case not supported, got {0} number of inputs, expected 3",
                                  inputs.size());
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_embedding_bag_packed_sum"},
                                         {"single_shave_embedding_bag_packed_sum.cpp"}};
            })
            .Case<VPU::GRUSequenceOp>([&](VPU::GRUSequenceOp gru) {
                const auto mode = gru.direction();
                VPUX_THROW_UNLESS(
                        mode == IE::RNNSequenceDirection::FORWARD || mode == IE::RNNSequenceDirection::REVERSE,
                        "GRUSequence supports FORWARD and REVERSE");
                mlir::IntegerAttr directionModeAttr =
                        (mode == IE::RNNSequenceDirection::FORWARD) ? getIntAttr(ctx, 0) : getIntAttr(ctx, 1);
                const auto shouldLinearBeforeReset =
                        static_cast<int64_t>(gru.should_linear_before_resetAttr() != nullptr);
                const auto shouldLinearBeforeResetAttr = getIntAttr(ctx, shouldLinearBeforeReset);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gru.hidden_sizeAttr(), directionModeAttr, gru.seq_lengthAttr(),
                                                     shouldLinearBeforeResetAttr, gru.clipAttr()},
                        {"single_shave_gru_sequence"},
                        {"single_shave_gru_sequence.cpp"}};
            })
            .Case<VPU::GRUSequenceFirstPartOp>([&](VPU::GRUSequenceFirstPartOp gru) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gru.hidden_sizeAttr(), gru.seq_lengthAttr(), gru.clipAttr()},
                        {"single_shave_gru_sequence_first_part"},
                        {"single_shave_gru_sequence_first_part.cpp"}};
            })
            .Case<VPU::GRUSequenceLastPartOp>([&](VPU::GRUSequenceLastPartOp gru) {
                const auto mode = static_cast<int64_t>(gru.direction() == IE::RNNSequenceDirection::FORWARD ? 0 : 1);
                const auto directionModeAttr = getIntAttr(ctx, mode);
                const auto shouldLinearBeforeReset =
                        static_cast<int64_t>(gru.should_linear_before_resetAttr() != nullptr);
                const auto shouldLinearBeforeResetAttr = getIntAttr(ctx, shouldLinearBeforeReset);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gru.hidden_sizeAttr(), directionModeAttr, gru.seq_lengthAttr(),
                                                     shouldLinearBeforeResetAttr, gru.clipAttr()},
                        {"single_shave_gru_sequence_last_part"},
                        {"single_shave_gru_sequence_last_part.cpp"}};
            })
            .Case<VPU::LSTMCellOp>([&](VPU::LSTMCellOp LSTMCell) {
                const auto inputDataShape = LSTMCell.inputData().getType().cast<mlir::ShapedType>().getShape();
                const auto batchSize = inputDataShape[0];
                const auto RNNForward = getIntAttr(ctx, 1);
                const auto nCells = getIntAttr(ctx, 1);
                const auto nBatch = getIntAttr(ctx, static_cast<int32_t>(batchSize));
                const auto useCellState = getIntAttr(ctx, 1);
                const auto outputsNumber = getIntAttr(ctx, 2);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{RNNForward, nCells, nBatch, useCellState, outputsNumber},
                        {"lstm_cell"},
                        {"lstm_cell.cpp"}};
            })
            .Case<VPU::LSTMGatesOp>([&](VPU::LSTMGatesOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"lstm_gates"}, {"lstm_gates.cpp"}};
            })
            .Case<VPU::LSTMSequenceOp>([&](VPU::LSTMSequenceOp LSTMSequence) {
                const auto direction = LSTMSequence.direction() == IE::RNNSequenceDirection::FORWARD ? 1 : 0;
                const auto RNNForward = getIntAttr(ctx, direction);
                const auto nCells = LSTMSequence.sequenceLengthAttr();
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{RNNForward, nCells},
                                         {"lstm_sequence"},
                                         {"lstm_sequence.cpp"}};
            })
            .Case<VPU::CTCGreedyDecoderSeqLenOp>([&](VPU::CTCGreedyDecoderSeqLenOp op) {
                const auto mergeRepeated = static_cast<int64_t>(op.mergeRepeatedAttr() != nullptr);
                const auto mergeRepeatedAttr = getIntAttr(ctx, mergeRepeated);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mergeRepeatedAttr},
                                         {"single_shave_ctc_greedy_decoder_seq_len"},
                                         {"single_shave_ctc_greedy_decoder_seq_len.cpp"}};
            })
            .Case<VPU::EmbeddingSegmentsSumOp>([&](VPU::EmbeddingSegmentsSumOp op) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{op.num_segments_valueAttr(), op.default_index_valueAttr()},
                        {"single_shave_embedding_segments_sum"},
                        {"single_shave_embedding_segments_sum.cpp"}};
            })
            .Case<VPU::SquaredDifferenceOp>([&](VPU::SquaredDifferenceOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_squared_difference_fp16"},
                                         {"eltwise_squared_difference_fp16.cpp"}};
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp op) {
                auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.stridesAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_beginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_endAttr());

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

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, padsBeginAttr, padsEndAttr},
                        {"single_shave_max_pool"},
                        {"single_shave_max_pool.cpp"}};
            })
            .Case<VPU::AbsOp>([&](VPU::AbsOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"abs_fp16"}, {"abs_fp16.cpp"}};
            })
            .Case<VPU::GeluOp>([&](VPU::GeluOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"gelu_fp16"}, {"gelu_fp16.cpp"}};
            })

            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp op) {
                auto group = getIntAttr(ctx, checked_cast<int32_t>(1));
                auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
                auto strides = parseIntArrayAttr<int64_t>(op.strides());
                auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto dilationsAttr = getIntArrayAttr(ctx, dilations);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{stridesAttr, padsBeginAttr, padsEndAttr, dilationsAttr, group},
                        {"single_shave_convolution"},
                        {"single_shave_convolution.cpp"}};
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp op) {
                auto group = static_cast<int64_t>(op.groups().getValue());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
                auto strides = parseIntArrayAttr<int64_t>(op.strides());
                auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto dilationsAttr = getIntArrayAttr(ctx, dilations);
                const auto groupAttr = getIntAttr(ctx, group);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{stridesAttr, padsBeginAttr, padsEndAttr, dilationsAttr, groupAttr},
                        {"single_shave_convolution"},
                        {"single_shave_convolution.cpp"}};
            })

            .Case<VPU::DFTOp>([&](VPU::DFTOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.input(), op.axes_attr());
                auto noAxes = op.axes_attr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"dft"}, {"dft.cpp"}};
            })
            .Case<VPU::RDFTOp>([&](VPU::RDFTOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.input(), op.axes_attr());
                auto noAxes = op.axes_attr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"rdft"}, {"rdft.cpp"}};
            })
            .Case<VPU::IDFTOp>([&](VPU::IDFTOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.input(), op.axes_attr());
                auto noAxes = op.axes_attr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"idft"}, {"idft.cpp"}};
            })
            .Case<VPU::IRDFTOp>([&](VPU::IRDFTOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.input(), op.axes_attr());
                auto noAxes = op.axes_attr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"irdft"}, {"irdft.cpp"}};
            })

            .Default([](mlir::Operation* unknownOp) -> VPUIP::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}

#undef CASE_REDUCE

VPUIP::KernelInfo SwKernelOp::getDummyKernelInfo() {
    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"dummy"}, {"dummy.cpp"}};
}

}  // namespace VPUIP
}  // namespace vpux
