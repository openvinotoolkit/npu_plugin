//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/bit.h>
#define REGION_YOLO_MAX_MASK_SIZE 9   // max mask size for region yolo op
#define PROPOSAL_MAX_RATIO 3          // max ratio size for proposal op
#define PROPOSAL_MAX_SCALE 6          // max scale size for proposal op
#define MAX_AXES_DIMS 4               // max axes size for reduce ops
#define SPACETOBATCH_MAX_INPUT_DIM 5  // max input size for spacetobatch op

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

uint64_t getFloatBits(float16 val) {
    return static_cast<uint64_t>(val.to_bits());
}

uint64_t getFloatBits(float val) {
    uint32_t f32Bits = llvm::bit_cast<uint32_t>(val);
    return static_cast<uint64_t>(f32Bits);
}

template <class IT, class OT>
void packAsFpIntoU64(const SmallVector<IT>& values, SmallVector<int64_t>& params) {
    static constexpr uint32_t PACKED_VALUES_COUNT = sizeof(int64_t) / sizeof(OT);
    static constexpr uint64_t bitWidth = sizeof(OT) * CHAR_BIT;
    OT fltValue[PACKED_VALUES_COUNT];
    size_t packIdx = 0;

    auto pack = [](OT fltVals[PACKED_VALUES_COUNT]) -> uint64_t {
        uint64_t ret = 0;
        for (uint32_t i = 0; i < PACKED_VALUES_COUNT; i++) {
            ret |= getFloatBits(fltVals[i]) << (bitWidth * i);
        }
        return ret;
    };

    for (const auto val : values) {
        fltValue[packIdx++] = static_cast<OT>(val);
        if (packIdx == PACKED_VALUES_COUNT) {
            params.push_back(pack(fltValue));
            packIdx = 0;  // reset pack index
        }
    }

    // Store trailing elements
    if (packIdx) {
        // Pad with zeros up to U64 alignment
        while (packIdx < PACKED_VALUES_COUNT) {
            fltValue[packIdx++] = 0;
        }
        params.push_back(pack(fltValue));
    }
}

void getQuantParamsAttr(mlir::MLIRContext* ctx, mlir::Type qType, mlir::Type pType, mlir::ArrayAttr& paramsAttr) {
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

    typedef decltype(scales)::value_type TS;
    typedef decltype(zeroes)::value_type TZ;

    // Convert & pack float values into u64 words for serialization
    llvm::SmallVector<int64_t> params;
    params.push_back(scales.size());
    if (pType.isF16()) {
        packAsFpIntoU64<TS, float16>(scales, params);
        packAsFpIntoU64<TZ, float16>(zeroes, params);
    } else if (pType.isF32()) {
        packAsFpIntoU64<TS, float>(scales, params);
        packAsFpIntoU64<TZ, float>(zeroes, params);
    } else {
        pType.dump();
        VPUX_THROW("Supported non-quantized type : f16/f32");
    }
    paramsAttr = getIntArrayAttr(ctx, std::move(params));
}

}  // namespace

namespace vpux {
namespace VPUIP {

void vpux::VPUIP::SwKernelOp::print(mlir::OpAsmPrinter& p) {
    p.printOptionalAttrDict(
            getOperation()->getAttrs(),
            /*elidedAttrs=*/{getOperandSegmentSizesAttrName().strref(), getKernelFunctionAttrName().strref(),
                             getTileIndexAttrName().strref(), getStridesAttrName().strref()});
    p << ' ';
    p.printAttributeWithoutType(getKernelFunctionAttr());

    auto& opBody = getBody();

    if (!opBody.empty()) {
        auto* entry = &opBody.front();

        unsigned opIdx = 0;
        printGroupOfOperands(p, entry, "inputs", getInputs(), opIdx);
        printGroupOfOperands(p, entry, "outputs", getOutputBuffs(), opIdx);
    }

    auto profData = getProfilingData();
    if (profData) {
        p << ' ' << "profiling_data";
        p << "(";
        p << profData;
        p << ' ' << ":";
        p << ' ';
        p << ArrayRef<mlir::Type>(profData.getType());
        p << ")";
    }

    auto opStrides = getStrides();
    if (opStrides) {
        p << ' ' << "strides";
        p << "(";
        p << opStrides;
        p << ")";
    }

    if (getTileIndex().has_value()) {
        p << ' ' << "on";
        p << ' ' << "tile";
        p << ' ';
        p.printAttributeWithoutType(getTileIndexAttr());
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
        if (parseResult.has_value()) {
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
        if (parseResult.has_value()) {
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
        if (!parseResult.has_value() || failed(*parseResult)) {
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
        if (parseResult.has_value() && failed(*parseResult)) {
            return mlir::failure();
        }
    }

    // Add derived `operandSegmentSizes` attribute based on parsed operands.
    auto operandSegmentSizes = parser.getBuilder().getDenseI32ArrayAttr(
            {inCount, outCount, static_cast<int32_t>(profiling_dataOperands.size())});
    result.addAttribute("operandSegmentSizes", operandSegmentSizes);

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
    mlir::Value profilingOutput = nullptr;

    build(builder, opState, inputs, results, profilingOutput, kernelFunction, tileIndex);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex,
                       mlir::ArrayAttr stride) {
    mlir::Value profilingOutput = nullptr;
    build(builder, opState, results.getTypes(), nullptr, kernelFunction, inputs, results, profilingOutput, tileIndex,
          stride, /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::Value profilingOutput, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex) {
    build(builder, opState, results.getTypes(), (profilingOutput ? profilingOutput.getType() : nullptr), kernelFunction,
          inputs, results, profilingOutput, tileIndex, /*stride*/ nullptr, /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::Value profilingOutput, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex, mlir::ArrayAttr stride) {
    build(builder, opState, results.getTypes(), (profilingOutput ? profilingOutput.getType() : nullptr), kernelFunction,
          inputs, results, profilingOutput, tileIndex, stride, /*profilingMetadata*/ nullptr);
}

mlir::LogicalResult SwKernelOp::verify() {
    if (!VPUIP::isCacheHandlingOp(*this)) {
        return mlir::success();
    }
    const auto op = getOperation();
    if (!op->getOperands().empty()) {
        return errorAt(op, "SW Kernel Cache Op should have no operands");
    }
    if (!op->getResults().empty()) {
        return errorAt(op, "SW Kernel Cache Op should have no results");
    }
    auto kernelFunc = op->getParentOfType<mlir::ModuleOp>().lookupSymbol<mlir::func::FuncOp>(getKernelFunctionAttr());
    if (kernelFunc.getFunctionType().getNumInputs() != 0) {
        return errorAt(op, "SW Kernel Cache Op func should have no inputs");
    }

    if (kernelFunc.getFunctionType().getNumResults() != 0) {
        return errorAt(op, "SW Kernel Cache Op func should have no results");
    }

    return mlir::success();
}

mlir::LogicalResult SwKernelOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                 mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                 mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                 mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::SwKernelOpAdaptor swKernelOp(operands, attrs);
    if (mlir::failed(swKernelOp.verify(loc))) {
        return mlir::failure();
    }

    for (auto out : swKernelOp.getOutputBuffs()) {
        inferredTypes.push_back(out.getType());
    }

    if (swKernelOp.getProfilingData() != nullptr) {
        inferredTypes.push_back(swKernelOp.getProfilingData().getType());
    }

    return mlir::success();
}

#define CASE_REDUCE(_OP_, _STR1_, _STR2_, _CTX_)                                                           \
    .Case<_OP_>([&](_OP_ reduce) {                                                                         \
        const auto keepDims = static_cast<int64_t>(reduce.getKeepDimsAttr() != nullptr);                   \
        const auto keepDimsAttr = getIntAttr(_CTX_, keepDims);                                             \
        const auto axesValue_size = static_cast<int64_t>(reduce.getAxesValueAttr().getValue().size());     \
        VPUX_THROW_UNLESS(axesValue_size <= MAX_AXES_DIMS, "Axes size {0} bigger than max axis size {1}",  \
                          axesValue_size, MAX_AXES_DIMS);                                                  \
        const auto axesValue_sizeAttr = getIntAttr(_CTX_, axesValue_size);                                 \
        return VPUIP::KernelInfo{                                                                          \
                SmallVector<mlir::Attribute>{keepDimsAttr, axesValue_sizeAttr, reduce.getAxesValueAttr()}, \
                {_STR1_},                                                                                  \
                {_STR2_}};                                                                                 \
    })

VPUIP::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    mlir::MLIRContext* ctx = origOp->getContext();

    return llvm::TypeSwitch<mlir::Operation*, VPUIP::KernelInfo>(origOp)
            .Case<VPU::ExpOp>([&](VPU::ExpOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"exp_fp16"}, {"exp_fp16.cpp"}};
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp gather) {
                const auto axisParam = computeReverseMemDim(gather.getInput(), gather.getAxisValue().value());
                const auto axisParamAttr = getIntAttr(gather.getContext(), axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, gather.getBatchDimsAttr()},
                                         {"single_shave_gather"},
                                         {"single_shave_gather.cpp"}};
            })
            .Case<VPU::GatherElementsOp>([&](VPU::GatherElementsOp gatherElements) {
                const auto axisParam = computeReverseMemDim(gatherElements.getInput(), gatherElements.getAxis());
                const auto axisParamAttr = getIntAttr(gatherElements.getContext(), axisParam);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"single_shave_gather_elements"},
                                         {"single_shave_gather_elements.cpp"}};
            })
            .Case<VPU::GatherNDOp>([&](VPU::GatherNDOp gatherND) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{gatherND.getBatchDimsAttr()},
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
                        computeReverseMemDim(scatterUpdate.getInput(), scatterUpdate.getAxisValue().value());
                const auto axisParamAttr = getIntAttr(scatterUpdate.getContext(), axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"single_shave_scatter_update"},
                                         {"single_shave_scatter_update.cpp"}};
            })
            .Case<VPU::ScatterElementsUpdateOp>([&](VPU::ScatterElementsUpdateOp scatterElementsUpdate) {
                const auto axisParam =
                        computeReverseMemDim(scatterElementsUpdate.getInput(), scatterElementsUpdate.getAxis());
                const auto axisParamAttr = getIntAttr(scatterElementsUpdate.getContext(), axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"single_shave_scatter_elements_update"},
                                         {"single_shave_scatter_elements_update.cpp"}};
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
                        SmallVector<mlir::Attribute>{hardsigmoid.getAlphaValueAttr(), hardsigmoid.getBetaValueAttr()},
                        {"activation_hardsigmoid"},
                        {}};
            })
            .Case<VPU::GridSampleOp>([&](VPU::GridSampleOp gridSample) {
                const auto alignCorners = static_cast<int64_t>(gridSample.getAlignCornersAttr() != nullptr);
                const auto alignCornersAttr = getIntAttr(ctx, alignCorners);

                int64_t mode = 0;
                if (gridSample.getModeAttr() != nullptr) {
                    mode = static_cast<int64_t>(gridSample.getModeAttr().getValue());
                }
                const auto modeIntAttr = getIntAttr(ctx, mode);

                int64_t paddingMode = 0;
                if (gridSample.getPaddingModeAttr() != nullptr) {
                    paddingMode = static_cast<int64_t>(gridSample.getPaddingModeAttr().getValue());
                }
                const auto paddingModeAttr = getIntAttr(ctx, paddingMode);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{alignCornersAttr, modeIntAttr, paddingModeAttr},
                                         {"single_shave_grid_sample"},
                                         {"single_shave_grid_sample.cpp"}};
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp softmax) {
                // input tensor, to transform axis
                const auto axisParam = computeReverseMemDim(softmax.getInput(), softmax.getAxisInd());
                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const int64_t padSize = 0;
                auto padSizeAttr = getIntAttr(ctx, padSize);
                if (softmax.getPadSize().has_value()) {
                    padSizeAttr = softmax.getPadSizeAttr();
                }
                const auto iType = softmax.getInput().getType().cast<vpux::NDTypeInterface>();
                if (iType.getElementType().isF32()) {
                    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, padSizeAttr},
                                             {"singleShaveSoftmaxFp32"},
                                             {"singleShaveSoftmaxFp32.cpp"}};
                } else {
                    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, padSizeAttr},
                                             {"singleShaveSoftmax"},
                                             {"singleShaveSoftmax.cpp"}};
                }
            })
            .Case<VPU::LogSoftmaxOp>([&](VPU::LogSoftmaxOp logSoftmax) {
                const auto axisParam = computeReverseMemDim(logSoftmax.getInput(), logSoftmax.getAxisInd());
                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr},
                                         {"singleShaveLogSoftmax"},
                                         {"single_shave_log_softmax.cpp"}};
            })
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolate) {
                const auto mode = static_cast<int64_t>(interpolate.getAttr().getMode().getValue());
                const auto coordMode = static_cast<int64_t>(interpolate.getAttr().getCoordMode().getValue());
                const auto nearestMode = static_cast<int64_t>(interpolate.getAttr().getNearestMode().getValue());
                const auto antialias = static_cast<int64_t>(interpolate.getAttr().getAntialias().getValue());
                const auto inOrder = DimsOrder::fromValue(interpolate.getInput());
                const auto initialInputDim = interpolate.getInitialInputDimsAttr().value();
                const auto initialInputDimsParam = permuteIntArrayAttr(inOrder, initialInputDim);
                const auto initialOutputDim = interpolate.getInitialOutputDimsAttr().value();
                const auto initialOutputDimsParam = permuteIntArrayAttr(inOrder, initialOutputDim);
                const auto tileFPList = parseFPArrayAttr<double>(interpolate.getTileOffsetAttrAttr());
                const auto cubeCoeffParam = interpolate.getAttr().getCubeCoeff().getValueAsDouble();
                const auto axisParam = parseIntArrayAttr<int64_t>(interpolate.getAxesAttrAttr());
                // Check the scaling dim size since the swkernel only support scaling at most two dims
                SmallVector<int64_t> scalingAxis;
                for (auto axis : axisParam) {
                    if (initialInputDim[checked_cast<unsigned>(axis)] !=
                        initialOutputDim[checked_cast<unsigned>(axis)]) {
                        scalingAxis.push_back(axis);
                    }
                }
                VPUX_THROW_WHEN(scalingAxis.size() > 2, "Unsupported scaling dim size {0} at '{1}'", scalingAxis.size(),
                                interpolate->getLoc());

                const auto initialInputOffset =
                        interpolate.getInitialInputOffsetAttr().has_value()
                                ? permuteIntArrayAttr(inOrder, interpolate.getInitialInputOffsetAttr().value())
                                : SmallVector<int64_t>(inOrder.numDims(), 0);
                const auto initialOutputOffset =
                        interpolate.getInitialOutputOffsetAttr().has_value()
                                ? permuteIntArrayAttr(inOrder, interpolate.getInitialOutputOffsetAttr().value())
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
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{elu.getXAttr()}, {"elu_fp16"}, {"elu_fp16.cpp"}};
            })
            .Case<VPU::ClampOp>([&](VPU::ClampOp clamp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{clamp.getMinAttr(), clamp.getMaxAttr()},
                                         {"clamp_fp16"},
                                         {"clamp_fp16.cpp"}};
            })
            .Case<VPU::FullyConnectedOp>([&](VPU::FullyConnectedOp fullyConnected) {
                const auto inputs = fullyConnected.getInputs();
                const auto inOrder = DimsOrder::fromValue(fullyConnected.getInput());

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
            .Case<VPU::ModOp>([&](VPU::ModOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_mod"}, {"eltwise_mod.cpp"}};
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
                const auto iType = mvn.getInput().getType().cast<vpux::NDTypeInterface>();
                const auto oType = mvn.getOutput().getType().cast<vpux::NDTypeInterface>();
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

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mvn.getAcrossChannelsAttr(),
                                                                      mvn.getNormalizeVarianceAttr(), mvn.getEpsAttr()},
                                         {"singleShaveMVN"},
                                         {"singleShaveMVN.cpp"}};
            })
            .Case<VPU::MVN1SumOp>([&](VPU::MVN1SumOp mvnSum) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{mvnSum.getAcrossChannelsAttr(), mvnSum.getNormalizeVarianceAttr()},
                        {"mvn1_sum"},
                        {"mvn1_sum.cpp"}};
            })
            .Case<VPU::MVN1MeanVarOp>([&](VPU::MVN1MeanVarOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getOrigShapeAttr(), op.getAcrossChannelsAttr(),
                                                                      op.getNormalizeVarianceAttr(), op.getEpsAttr()},
                                         {"mvn1_mean_var"},
                                         {"mvn1_mean_var.cpp"}};
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{op.getAcrossChannelsAttr(), op.getNormalizeVarianceAttr()},
                        {"mvn1_norm"},
                        {"mvn1_norm.cpp"}};
            })
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op mvn) {
                mlir::MLIRContext* ctx = mvn.getContext();
                // Convert 'axes' to physical reversed (innermost first) equivalent
                const auto axesVal = parseIntArrayAttr<int64_t>(mvn.getAxes());
                SmallVector<int64_t> memAxes;
                for (const auto axis : axesVal) {
                    memAxes.push_back(computeReverseMemDim(mvn.getInput(), axis));
                }
                const auto numAxes = memAxes.size();
                const auto epsMode = static_cast<int64_t>(mvn.getEpsModeAttr().getValue());
                const auto memAxesAttr = getIntArrayAttr(ctx, memAxes);
                const auto numAxesAttr = getIntAttr(ctx, numAxes);
                const auto epsModeAttr = getIntAttr(ctx, epsMode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mvn.getNormalizeVarianceAttr(), epsModeAttr,
                                                                      mvn.getEpsAttr(), numAxesAttr, memAxesAttr},
                                         {"mvn6"},
                                         {"mvn6.cpp"}};
            })
            .Case<VPU::MemPermuteOp>([&](VPU::MemPermuteOp permute) {
                auto memPermArr = reversePermutation(permute.getMemPerm());
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntArrayAttr(ctx, memPermArr)},
                                         {"reorder_fp16"},
                                         {"reorder_fp16.cpp"}};
            })
            .Case<VPU::LRNOp>([&](VPU::LRNOp LRN) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{LRN.getAlphaAttr(), LRN.getBetaAttr(),
                                                                      LRN.getBiasAttr(), LRN.getSizeAttr()},
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
                const auto maskSize = static_cast<int64_t>(regionYolo.getMaskAttr().getValue().size());
                const auto maskSizeParamAttr = getIntAttr(ctx, maskSize);
                // Supplement for maskAttrInput to meet array setSize(9) defined in kernel
                const auto maskVec = parseIntArrayAttr<int64_t>(regionYolo.getMaskAttr());
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

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{regionYolo.getCoordsAttr(), regionYolo.getClassesAttr(),
                                                     regionYolo.getNumRegionsAttr(), regionYolo.getDoSoftmaxAttr(),
                                                     maskSizeParamAttr, maskAttrInput, regionYolo.getAxisAttr(),
                                                     regionYolo.getEndAxisAttr()},
                        {"single_shave_region_yolo"},
                        {"single_shave_region_yolo.cpp"}};
            })
            .Case<VPU::TopKOp>([&](VPU::TopKOp topk) {
                const auto axisParam = computeReverseMemDim(topk.getInput(), topk.getAxis());
                const auto mode = static_cast<int64_t>(topk.getModeAttr().getValue());
                const auto sortMode = static_cast<int64_t>(topk.getSortAttr().getValue());
                const auto kValue = static_cast<int64_t>(topk.getKValue().value());

                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const auto modeIntAttr = getIntAttr(ctx, mode);
                const auto sortModeAttr = getIntAttr(ctx, sortMode);
                const auto kValueAttr = getIntAttr(ctx, kValue);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{axisParamAttr, modeIntAttr, sortModeAttr, kValueAttr},
                        {"single_shave_topk"},
                        {"single_shave_topk.cpp"}};
            })
            .Case<VPU::ExtractImagePatchesOp>([&](VPU::ExtractImagePatchesOp op) {
                auto sizes = parseIntArrayAttr<int64_t>(op.getSizesAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto rates = parseIntArrayAttr<int64_t>(op.getRatesAttr());
                const auto autoPad = static_cast<int32_t>(op.getAutoPadAttr().getValue());

                const auto iType = op.getData().getType().cast<vpux::NDTypeInterface>();
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
                const auto inOrder = DimsOrder::fromValue(pad.getInput());
                const auto padBegin = permuteIntArrayAttr(inOrder, pad.getPadsBeginAttr().value());
                const auto padEnd = permuteIntArrayAttr(inOrder, pad.getPadsEndAttr().value());
                const auto padMode = static_cast<int64_t>(pad.getModeAttr().getValue());

                const auto padBeginAttr = getIntArrayAttr(ctx, padBegin);
                const auto padEndAttr = getIntArrayAttr(ctx, padEnd);
                const auto padModeAttr = getIntAttr(ctx, padMode);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{padBeginAttr, padEndAttr, pad.getPadValueAttrAttr(), padModeAttr},
                        {"single_shave_pad"},
                        {"single_shave_pad.cpp"}};
            })
            .Case<VPU::AvgPoolOp>([&](VPU::AvgPoolOp op) {
                constexpr size_t MAX_ATTR_SZ = 3;  // base on filter description
                auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBeginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEndAttr());
                const auto excludePads = static_cast<int64_t>(op.getExcludePadsAttr() != nullptr);
                // In order to support any format, pad with identity up to the max attribute size (3).
                // Filter definition have a strict dependency between input tensor rank, params size and layout, but for
                // hw usage we can cover and situation when this are not respected.
                for (auto i = kernelSize.size(); i < MAX_ATTR_SZ; i++) {
                    kernelSize.insert(kernelSize.begin(), 1);
                    strides.insert(strides.begin(), 1);
                    padsBegin.insert(padsBegin.begin(), 0);
                    padsEnd.insert(padsEnd.begin(), 0);
                }
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
            .Case<VPU::AdaptiveAvgPoolOp>([&](VPU::AdaptiveAvgPoolOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_adaptive_pool"},
                                         {"single_shave_adaptive_pool.cpp"}};
            })
            .Case<VPU::FakeQuantizeOp>([&](VPU::FakeQuantizeOp op) {
                const auto iType = op.getInput().getType().cast<vpux::NDTypeInterface>();
                const auto oType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getElementType().isF16() && oType.getElementType().isF16(),
                                  "Only supports FP16 in/out");
                const auto levelsAttr = getIntAttr(ctx, op.getLevels());
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{levelsAttr},
                                         {"fake_quantize"},
                                         {"fake_quantize.cpp"}};
            })
            .Case<VPU::QuantizeOp>([&](VPU::QuantizeOp op) {
                const auto iType = op.getInput().getType().cast<vpux::NDTypeInterface>();
                const auto oType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
                mlir::ArrayAttr paramsAttr;
                getQuantParamsAttr(ctx, oType.getElementType(), iType.getElementType(), paramsAttr);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{paramsAttr}, {"quantize"}, {"quantize.cpp"}};
            })
            .Case<VPU::DequantizeOp>([&](VPU::DequantizeOp op) {
                const auto iType = op.getInput().getType().cast<vpux::NDTypeInterface>();
                const auto oType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
                mlir::ArrayAttr paramsAttr;
                getQuantParamsAttr(ctx, iType.getElementType(), oType.getElementType(), paramsAttr);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{paramsAttr}, {"dequantize"}, {"dequantize.cpp"}};
            })
            .Case<VPU::DynamicQuantizeOp>([&](VPU::DynamicQuantizeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"dynamic_quantize"},
                                         {"dynamic_quantize.cpp"}};
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp depth_to_space) {
                const auto mode = static_cast<int64_t>(depth_to_space.getModeAttr().getValue());
                const auto modeAttr = getIntAttr(ctx, mode);
                SmallVector<mlir::Attribute> paramAttr = {depth_to_space.getBlockSizeAttr(), modeAttr};
                if (depth_to_space.getPaddedChannels().has_value()) {
                    auto paddedIC = depth_to_space.getPaddedChannels().value().getInput();
                    auto paddedOC = depth_to_space.getPaddedChannels().value().getOutput();

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
                const auto mode = static_cast<int64_t>(space_to_depth.getModeAttr().getValue());
                const auto modeAttr = getIntAttr(ctx, mode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{space_to_depth.getBlockSizeAttr(), modeAttr},
                                         {"single_shave_space_to_depth"},
                                         {"single_shave_space_to_depth.cpp"}};
            })
            .Case<VPU::SpaceToBatch>([&](VPU::SpaceToBatch space_to_batch) {
                const auto blockShape = parseIntArrayAttr<int64_t>(space_to_batch.getBlockShapeValueAttr());
                const auto padsBegin = parseIntArrayAttr<int64_t>(space_to_batch.getPadsBeginValueAttr());
                const auto padsEnd = parseIntArrayAttr<int64_t>(space_to_batch.getPadsEndValueAttr());

                const auto inputDim = static_cast<int64_t>(space_to_batch.getBlockShapeValueAttr().getValue().size());

                int64_t blockShapeList[SPACETOBATCH_MAX_INPUT_DIM];
                int64_t padsBeginList[SPACETOBATCH_MAX_INPUT_DIM];
                int64_t padsEndList[SPACETOBATCH_MAX_INPUT_DIM];

                for (int i = 0; i < inputDim; i++) {
                    if (i >= inputDim) {
                        blockShapeList[i] = -1;
                        padsBeginList[i] = -1;
                        padsEndList[i] = -1;
                    } else {
                        blockShapeList[i] = blockShape[i];
                        padsBeginList[i] = padsBegin[i];
                        padsEndList[i] = padsEnd[i];
                    }
                }

                const auto blockShapeArryRef = ArrayRef<int64_t>(blockShapeList, SPACETOBATCH_MAX_INPUT_DIM);
                const auto padsBeginArryRef = ArrayRef<int64_t>(padsBeginList, SPACETOBATCH_MAX_INPUT_DIM);
                const auto padsEndArryRef = ArrayRef<int64_t>(padsEndList, SPACETOBATCH_MAX_INPUT_DIM);
                const auto blockShapeAttr = getIntArrayAttr(ctx, blockShapeArryRef);
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBeginArryRef);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEndArryRef);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{blockShapeAttr, padsBeginAttr, padsEndAttr},
                                         {"space_to_batch"},
                                         {"space_to_batch.cpp"}};
            })
            .Case<VPU::SeluOp>([&](VPU::SeluOp selu) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{selu.getAlphaValueAttr(), selu.getLambdaValueAttr()},
                        {"selu_fp16"},
                        {"selu_fp16.cpp"}};
            })
            .Case<VPU::LeakyReluOp>([&](VPU::LeakyReluOp leakyRelu) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{leakyRelu.getNegativeSlopeAttr()},
                                         {"leaky_relu_fp16"},
                                         {"leaky_relu_fp16.cpp"}};
            })
            .Case<VPU::SwishOp>([&](VPU::SwishOp swish) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{swish.getBetaValueAttr()},
                                         {"swish_fp16"},
                                         {"swish_fp16.cpp"}};
            })
            .Case<VPU::ReLUOp>([&](VPU::ReLUOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_relu"}, {}};
            })
            .Case<VPU::NegativeOp>([&](VPU::NegativeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_negative"},
                                         {"single_shave_negative.cpp"}};
            })
            .Case<VPU::StridedSliceOp>([&](VPU::StridedSliceOp stridedslice) {
                mlir::MLIRContext* ctx = origOp->getContext();

                const auto stridedSliceBegins = padIntArrayAttr(stridedslice.getBeginsAttr());
                const auto stridedSliceEnds = padIntArrayAttr(stridedslice.getEndsAttr());
                const auto stridedSliceStrides = padIntArrayAttr(stridedslice.getStridesAttr().value());

                const auto stridedSliceBeginsAttr = getIntArrayAttr(ctx, stridedSliceBegins);
                const auto stridedSliceEndsAttr = getIntArrayAttr(ctx, stridedSliceEnds);
                const auto stridedSliceStridesAttr = getIntArrayAttr(ctx, stridedSliceStrides);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{stridedSliceBeginsAttr, stridedSliceEndsAttr,
                                                                      stridedSliceStridesAttr},
                                         {"single_shave_stridedslice"},
                                         {"single_shave_stridedslice.cpp"}};
            })
            .Case<VPU::ReverseSequenceOp>([&](VPU::ReverseSequenceOp ReverseSequence) {
                const auto batchAxis = computeReverseMemDim(ReverseSequence.getData(), ReverseSequence.getBatchAxis());
                const auto seqAxis = computeReverseMemDim(ReverseSequence.getData(), ReverseSequence.getSeqAxis());
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{getIntAttr(ctx, batchAxis), getIntAttr(ctx, seqAxis)},
                        {"single_shave_reverse_sequence"},
                        {"single_shave_reverse_sequence.cpp"}};
            })
            .Case<VPU::YuvToRgbOp>([&](VPU::YuvToRgbOp yuvToRgb) {
                mlir::MLIRContext* ctx = origOp->getContext();
                const auto inFmt = yuvToRgb.getInFmtAttr().getValue();
                const auto outFmt = static_cast<int64_t>(yuvToRgb.getOutFmtAttr().getValue()) - 2;
                const auto outFmtIntAttr = getIntAttr(ctx, outFmt);
                auto singlePlane = (yuvToRgb.getInput2() == nullptr);

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
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{rand.getGlobalSeedAttr(), rand.getOpSeedAttr()},
                                         {"random_uniform"},
                                         {"random_uniform.cpp"}};
            })
            .Case<VPU::ROIPoolingOp>([&](VPU::ROIPoolingOp roi) {
                const auto iType = roi.getInput().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                const auto method = static_cast<int64_t>(roi.getMethodAttr().getValue());
                const auto methodAttr = getIntAttr(ctx, method);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{roi.getOutputSizeAttr(), roi.getSpatialScaleAttr(), methodAttr},
                        {"single_shave_roipooling"},
                        {"single_shave_roipooling.cpp"}};
            })
            .Case<VPU::ROIAlignOp>([&](VPU::ROIAlignOp roiAlign) {
                const auto iType = roiAlign.getInput().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                const auto pooledH = roiAlign.getPooledHAttr();
                const auto pooledW = roiAlign.getPooledWAttr();
                const auto samplingRatio = roiAlign.getSamplingRatioAttr();
                const auto spatialScale = roiAlign.getSpatialScaleAttr();

                const auto poolingMode = static_cast<int64_t>(roiAlign.getPoolingModeAttr().getValue());
                const auto poolingModeAttr = getIntAttr(ctx, poolingMode);

                const auto alignedMode = static_cast<int64_t>(roiAlign.getAlignedModeAttr().getValue());
                const auto alignedModeAttr = getIntAttr(ctx, alignedMode);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{pooledH, pooledW, samplingRatio, spatialScale,
                                                                      poolingModeAttr, alignedModeAttr},
                                         {"roi_align"},
                                         {}};
            })
            .Case<VPU::RollOp>([&](VPU::RollOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_roll"},
                                         {"single_shave_roll.cpp"}};
            })
            .Case<VPU::OneHotOp>([&](VPU::OneHotOp oneHot) {
                int64_t axis = oneHot.getAxis();
                const auto shape = getShape(oneHot.getInput());
                auto nDims = checked_cast<int64_t>(shape.size());
                const int64_t actualAxis = (axis < 0) ? -axis - 1 : nDims - axis;

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{getIntAttr(ctx, actualAxis), oneHot.getDepthAttr(),
                                                     oneHot.getOnValueAttr(), oneHot.getOffValueAttr()},
                        {"single_shave_onehot"},
                        {"single_shave_onehot.cpp"}};
            })
            .Case<VPU::ReorgYoloOp>([&](VPU::ReorgYoloOp reorgYolo) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{reorgYolo.getStrideAttr()},
                                         {"single_shave_reorg_yolo"},
                                         {"single_shave_reorg_yolo.cpp"}};
            })
            .Case<VPU::ProposalOp>([&](VPU::ProposalOp proposal) {
                const auto baseSizeAttr = proposal.getProposalAttrsAttr().getBaseSize();
                const auto preNmsTopNAttr = proposal.getProposalAttrsAttr().getPreNmsTopN();
                const auto postNmsTopNAttr = proposal.getProposalAttrsAttr().getPostNmsTopN();
                const auto nmsThreshAttr = proposal.getProposalAttrsAttr().getNmsThresh();
                const auto featStrideAttr = proposal.getProposalAttrsAttr().getFeatStride();
                const auto minSizeAttr = proposal.getProposalAttrsAttr().getMinSize();

                const auto ratioSize = static_cast<int>(proposal.getProposalAttrsAttr().getRatio().size());
                const auto ratio = parseFPArrayAttr<double>(proposal.getProposalAttrsAttr().getRatio());
                float ratioList[PROPOSAL_MAX_RATIO];
                for (int i = 0; i < PROPOSAL_MAX_RATIO; i++) {
                    if (i >= ratioSize) {
                        ratioList[i] = -1;
                    } else {
                        ratioList[i] = checked_cast<float>(ratio[i]);
                    }
                }
                const auto ratioRef = ArrayRef<float>(ratioList, PROPOSAL_MAX_RATIO);
                const auto ratioAttr = getFPArrayAttr(ctx, ratioRef);

                const auto scaleSize = static_cast<int>(proposal.getProposalAttrsAttr().getScale().size());
                const auto scale = parseFPArrayAttr<double>(proposal.getProposalAttrsAttr().getScale());
                float scaleList[PROPOSAL_MAX_SCALE];
                for (int i = 0; i < PROPOSAL_MAX_SCALE; i++) {
                    if (i >= scaleSize) {
                        scaleList[i] = -1;
                    } else {
                        scaleList[i] = checked_cast<float>(scale[i]);
                    }
                }
                const auto scaleRef = ArrayRef<float>(scaleList, PROPOSAL_MAX_SCALE);
                const auto scaleAttr = getFPArrayAttr(ctx, scaleRef);

                const bool clipBeforeNms = proposal.getProposalAttrsAttr().getClipBeforeNms().getValue();
                const int64_t clipBeforeNmsInt = clipBeforeNms != 0;
                const auto clipBeforeNmsAttr = getIntAttr(ctx, clipBeforeNmsInt);

                const bool clipAfterNms = proposal.getProposalAttrsAttr().getClipAfterNms().getValue();
                const int64_t clipAfterNmsInt = clipAfterNms != 0;
                const auto clipAfterNmsAttr = getIntAttr(ctx, clipAfterNmsInt);

                const bool normalize = proposal.getProposalAttrsAttr().getNormalize().getValue();
                const int64_t normalizeInt = normalize != 0;
                const auto normalizeAttr = getIntAttr(ctx, normalizeInt);

                const auto boxSizeScaleAttr = proposal.getProposalAttrsAttr().getBoxSizeScale();
                const auto boxCoordinateScaleAttr = proposal.getProposalAttrsAttr().getBoxCoordinateScale();

                const auto framework = proposal.getProposalAttrsAttr().getFramework().getValue().str();
                int64_t frameworkInt = framework == "tensorflow";
                const auto frameworkAttr = getIntAttr(ctx, frameworkInt);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{baseSizeAttr, preNmsTopNAttr, postNmsTopNAttr, nmsThreshAttr,
                                                     featStrideAttr, minSizeAttr, ratioAttr, scaleAttr,
                                                     clipBeforeNmsAttr, clipAfterNmsAttr, normalizeAttr,
                                                     boxSizeScaleAttr, boxCoordinateScaleAttr, frameworkAttr},
                        {"single_shave_proposal"},
                        {"single_shave_proposal.cpp"}};
            })
            .Case<VPU::PSROIPoolingOp>([&](VPU::PSROIPoolingOp psroi) {
                const auto iType = psroi.getInput().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());
                VPUX_THROW_UNLESS(psroi.getMode() == IE::PSROIPoolingMode::AVERAGE,
                                  "Supporting only average psroi mode, got {0}", psroi.getMode());

                const auto spatialBinsX = static_cast<int64_t>(psroi.getSpatialBinsXAttr() != nullptr);
                const auto spatialBinsY = static_cast<int64_t>(psroi.getSpatialBinsYAttr() != nullptr);
                const auto mode = static_cast<int64_t>(psroi.getModeAttr() != nullptr);

                const auto spatialBinsXAttr = getIntAttr(ctx, spatialBinsX);
                const auto spatialBinsYAttr = getIntAttr(ctx, spatialBinsY);
                const auto modeAttr = getIntAttr(ctx, mode);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{psroi.getOutputDimAttr(), psroi.getSpatialScaleAttr(),
                                                     psroi.getGroupSizeAttr(), spatialBinsXAttr, spatialBinsYAttr,
                                                     modeAttr},
                        {"single_shave_ps_roipooling"},
                        {"single_shave_ps_roipooling.cpp"}};
            })
            .Case<VPU::NonMaxSuppressionOp>([&](VPU::NonMaxSuppressionOp nms) {
                const auto boxEncoding = static_cast<int64_t>(nms.getBoxEncodingAttr().getValue());
                const auto boxEncodingAttr = getIntAttr(ctx, boxEncoding);
                const auto sortResultDescending = static_cast<int64_t>(nms.getSortResultDescendingAttr() != nullptr);
                const auto sortResultDescendingAttr = getIntAttr(ctx, sortResultDescending);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{nms.getMaxOutputBoxesPerClassValueAttr(),
                                                     nms.getIouThresholdValueAttr(), nms.getScoreThresholdValueAttr(),
                                                     nms.getSoftNmsSigmaValueAttr(), boxEncodingAttr,
                                                     sortResultDescendingAttr},
                        {"nms_fp16"},
                        {"nms_fp16.cpp"}};
            })
            .Case<VPU::CTCGreedyDecoderOp>([&](VPU::CTCGreedyDecoderOp op) {
                const auto mergeRepeated = static_cast<int64_t>(op.getMergeRepeatedAttr() != nullptr);
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
                auto repeats_size = static_cast<int64_t>(tileOp.getRepeatsValuesAttr().getValue().size());
                auto repeats_sizeAttr = getIntAttr(ctx, repeats_size);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{repeats_sizeAttr, tileOp.getRepeatsValuesAttr()},
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
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getInputWidthAttr(), op.getInputHeightAttr()},
                                         {"detection_output_normalize"},
                                         {"detection_output_normalize.cpp"}};
            })
            .Case<VPU::DetectionOutputDecodeBoxesOp>([&](VPU::DetectionOutputDecodeBoxesOp op) {
                const auto codeType = [&]() -> int64_t {
                    enum CodeType : int64_t { CENTER_SIZE, CORNER, CORNER_SIZE };
                    switch (op.getCodeType()) {
                    case IE::DetectionOutputCodeType::CENTER_SIZE:
                        return CodeType::CENTER_SIZE;
                    case IE::DetectionOutputCodeType::CORNER:
                        return CodeType::CORNER;
                    case IE::DetectionOutputCodeType::CORNER_SIZE:
                        return CodeType::CORNER_SIZE;
                    }

                    const auto codeTypeString = stringifyDetectionOutputCodeType(op.getCodeType());
                    VPUX_THROW("Unsupported DetectionOutput codeType: {0}", codeTypeString);
                }();

                const auto clipBeforeNms = static_cast<int64_t>(op.getClipBeforeNms());
                const auto clipBeforeNmsAttr = getIntAttr(ctx, clipBeforeNms);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntAttr(ctx, codeType), clipBeforeNmsAttr},
                                         {"detection_output_decode_boxes"},
                                         {"detection_output_decode_boxes.cpp"}};
            })
            .Case<VPU::DetectionOutputSortTopKOp>([&](VPU::DetectionOutputSortTopKOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getConfidenceThresholdAttr(), op.getTopKAttr(),
                                                                      op.getBackgroundIdAttr()},
                                         {"detection_output_sort_top_k"},
                                         {"detection_output_sort_top_k.cpp"}};
            })
            .Case<VPU::DetectionOutputSelectBoxesOp>([&](VPU::DetectionOutputSelectBoxesOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"detection_output_select_boxes"},
                                         {"detection_output_select_boxes.cpp"}};
            })
            .Case<VPU::DetectionOutputNmsCaffeOp>([&](VPU::DetectionOutputNmsCaffeOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getNmsThresholdAttr()},
                                         {"detection_output_nms_caffe"},
                                         {"detection_output_nms_caffe.cpp"}};
            })
            .Case<VPU::DetectionOutputCollectResultsOp>([&](VPU::DetectionOutputCollectResultsOp op) {
                const auto clipAfterNms = static_cast<int64_t>(op.getClipAfterNms());
                const auto clipAfterNmsAttr = getIntAttr(ctx, clipAfterNms);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getKeepTopKAttr(), clipAfterNmsAttr},
                                         {"detection_output_collect_results"},
                                         {"detection_output_collect_results.cpp"}};
            })
            .Case<VPU::PermuteQuantizeOp>([&](VPU::PermuteQuantizeOp op) {
                // permutation params convert to shv order
                auto memPermArr = reversePermutation(op.getMemPerm());
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
                const auto iType = op.getInput().getType().cast<vpux::NDTypeInterface>();
                const auto oType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
                const auto inOrder = DimsOrder::fromValue(op.getInput());
                const auto outOrder = DimsOrder::fromValue(op.getOutput());
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
                const auto mode = static_cast<int64_t>(round.getModeAttr().getValue());
                const auto modeIntAttr = getIntAttr(ctx, mode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{modeIntAttr}, {"round_fp16"}, {"round_fp16.cpp"}};
            })
            .Case<VPU::LogOp>([&](VPU::LogOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_log"}, {}};
            })
            .Case<VPU::NormalizeL2Op>([&](VPU::NormalizeL2Op normalizeL2) {
                const auto epsMode = static_cast<int64_t>(normalizeL2.getEpsModeAttr().getValue());
                const auto epsModeAttr = getIntAttr(ctx, epsMode);
                auto axes = parseIntArrayAttr<int32_t>(normalizeL2.getAxesValueAttr());
                auto noAxes = axes.size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int32_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{normalizeL2.getEpsAttr(), epsModeAttr, noAxesAttr, axesAttr},
                        {"normalize_l2_fp16"},
                        {"normalize_l2_fp16.cpp"}};
            })
            .Case<VPU::CumSumOp>([&](VPU::CumSumOp cumSum) {
                const auto axisParam = computeReverseMemDim(cumSum.getInput(), cumSum.getAxisValue().value());
                const auto exclusive = static_cast<int64_t>(cumSum.getExclusiveAttr() != nullptr);
                const auto reverse = static_cast<int64_t>(cumSum.getReverseAttr() != nullptr);

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
            .Case<VPU::EmbeddingBagOffsetsSumOp>([&](VPU::EmbeddingBagOffsetsSumOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getDefaultIndexValueAttr()},
                                         {"single_shave_embedding_bag_offsets_sum"},
                                         {"single_shave_embedding_bag_offsets_sum.cpp"}};
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
            .Case<VPU::EyeOp>([&](VPU::EyeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"single_shave_eye"},
                                         {"single_shave_eye.cpp"}};
            })
            .Case<VPU::GRUSequenceOp>([&](VPU::GRUSequenceOp gru) {
                const auto mode = gru.getDirection();
                VPUX_THROW_UNLESS(
                        mode == IE::RNNSequenceDirection::FORWARD || mode == IE::RNNSequenceDirection::REVERSE,
                        "GRUSequence supports FORWARD and REVERSE");
                mlir::IntegerAttr directionModeAttr =
                        (mode == IE::RNNSequenceDirection::FORWARD) ? getIntAttr(ctx, 0) : getIntAttr(ctx, 1);
                const auto shouldLinearBeforeReset =
                        static_cast<int64_t>(gru.getShouldLinearBeforeResetAttr() != nullptr);
                const auto shouldLinearBeforeResetAttr = getIntAttr(ctx, shouldLinearBeforeReset);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gru.getHiddenSizeAttr(), directionModeAttr, gru.getSeqLengthAttr(),
                                                     shouldLinearBeforeResetAttr, gru.getClipAttr()},
                        {"single_shave_gru_sequence"},
                        {"single_shave_gru_sequence.cpp"}};
            })
            .Case<VPU::GRUSequenceFirstPartOp>([&](VPU::GRUSequenceFirstPartOp gru) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{gru.getHiddenSizeAttr(), gru.getSeqLengthAttr(),
                                                                      gru.getClipAttr()},
                                         {"single_shave_gru_sequence_first_part"},
                                         {"single_shave_gru_sequence_first_part.cpp"}};
            })
            .Case<VPU::GRUSequenceLastPartOp>([&](VPU::GRUSequenceLastPartOp gru) {
                const auto mode = static_cast<int64_t>(gru.getDirection() == IE::RNNSequenceDirection::FORWARD ? 0 : 1);
                const auto directionModeAttr = getIntAttr(ctx, mode);
                const auto shouldLinearBeforeReset =
                        static_cast<int64_t>(gru.getShouldLinearBeforeResetAttr() != nullptr);
                const auto shouldLinearBeforeResetAttr = getIntAttr(ctx, shouldLinearBeforeReset);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gru.getHiddenSizeAttr(), directionModeAttr, gru.getSeqLengthAttr(),
                                                     shouldLinearBeforeResetAttr, gru.getClipAttr()},
                        {"single_shave_gru_sequence_last_part"},
                        {"single_shave_gru_sequence_last_part.cpp"}};
            })
            .Case<VPU::LSTMCellOp>([&](VPU::LSTMCellOp LSTMCell) {
                const auto inputDataShape = LSTMCell.getInputData().getType().cast<mlir::ShapedType>().getShape();
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
                const auto direction = LSTMSequence.getDirection() == IE::RNNSequenceDirection::FORWARD ? 1 : 0;
                const auto RNNForward = getIntAttr(ctx, direction);
                const auto nCells = LSTMSequence.getSequenceLengthAttr();
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{RNNForward, nCells},
                                         {"lstm_sequence"},
                                         {"lstm_sequence.cpp"}};
            })
            .Case<VPU::CTCGreedyDecoderSeqLenOp>([&](VPU::CTCGreedyDecoderSeqLenOp op) {
                const auto mergeRepeated = static_cast<int64_t>(op.getMergeRepeatedAttr() != nullptr);
                const auto mergeRepeatedAttr = getIntAttr(ctx, mergeRepeated);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mergeRepeatedAttr},
                                         {"single_shave_ctc_greedy_decoder_seq_len"},
                                         {"single_shave_ctc_greedy_decoder_seq_len.cpp"}};
            })
            .Case<VPU::EmbeddingSegmentsSumOp>([&](VPU::EmbeddingSegmentsSumOp op) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{op.getNumSegmentsValueAttr(), op.getDefaultIndexValueAttr()},
                        {"single_shave_embedding_segments_sum"},
                        {"single_shave_embedding_segments_sum.cpp"}};
            })
            .Case<VPU::SquaredDifferenceOp>([&](VPU::SquaredDifferenceOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"eltwise_squared_difference"},
                                         {"eltwise_squared_difference.cpp"}};
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp op) {
                constexpr size_t MAX_ATTR_SZ = 3;  // base on filter description
                auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBeginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEndAttr());
                // In order to support any format, pad with identity up to the max attribute size (3).
                // Filter definition have a strict dependency between input tensor rank, params size and layout, but for
                // hw usage we can cover and situation when this are not respected.
                for (auto i = kernelSize.size(); i < MAX_ATTR_SZ; i++) {
                    kernelSize.insert(kernelSize.begin(), 1);
                    strides.insert(strides.begin(), 1);
                    padsBegin.insert(padsBegin.begin(), 0);
                    padsEnd.insert(padsEnd.begin(), 0);
                }
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
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_abs"}, {}};
            })
            .Case<VPU::GeluOp>([&](VPU::GeluOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"gelu_fp16"}, {"gelu_fp16.cpp"}};
            })
            .Case<VPU::BucketizeOp>([&](VPU::BucketizeOp bucketize) {
                const auto dataType = bucketize.getData().getType().cast<vpux::NDTypeInterface>();
                const auto bucketsType = bucketize.getBuckets().getType().cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(dataType.getElementType().isF16() && bucketsType.getElementType().isF16(),
                                  "Only supports FP16 for Input1 & Input2");
                const auto with_right_bound = static_cast<int64_t>(bucketize.getWithRightBound());
                const auto with_right_boundAttr = getIntAttr(ctx, with_right_bound);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{with_right_boundAttr},
                                         {"bucketize_fp16"},
                                         {"bucketize_fp16.cpp"}};
            })
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp op) {
                auto group = getIntAttr(ctx, checked_cast<int32_t>(1));
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
                auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
                auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
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
                auto group = static_cast<int64_t>(op.getGroups().value());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
                auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
                auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
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
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"dft"}, {"dft.cpp"}};
            })
            .Case<VPU::RDFTUncutOp>([&](VPU::RDFTUncutOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"rdft"}, {"rdft.cpp"}};
            })
            .Case<VPU::IDFTOp>([&](VPU::IDFTOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"idft"}, {"idft.cpp"}};
            })
            .Case<VPU::IRDFTLastAxisOp>([&](VPU::IRDFTLastAxisOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"irdft"}, {"irdft.cpp"}};
            })
            .Case<VPU::ConditionalCopyOp>([&](VPU::ConditionalCopyOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"conditionalCopy"}, {"conditionalCopy.cpp"}};
            })

            .Default([](mlir::Operation* unknownOp) -> VPUIP::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}

#undef CASE_REDUCE

VPUIP::KernelInfo SwKernelOp::getDummyKernelInfo() {
    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"dummy"}, {"dummy.cpp"}};
}

size_t SwKernelOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto swKernelOp = mlir::cast<VPUIP::SwKernelOp>(this->getOperation());
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: Expose API to get arch from cost model
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(calculateShaveActCycles(swKernelOp, costModel, arch));
}

}  // namespace VPUIP
}  // namespace vpux
