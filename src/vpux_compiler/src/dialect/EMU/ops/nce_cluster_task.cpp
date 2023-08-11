//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

//
// NCEClusterTaskOp::build
//

void EMU::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output,
                                  mlir::Value input, mlir::Value weights, mlir::Value weightsTable,
                                  VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
                                  mlir::ArrayAttr kernel_strides, mlir::ArrayAttr kernel_padding,
                                  Optional<VPU::PPETaskAttr> ppe) {
    build(builder, state, output, input, weights, weightsTable,
          VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type), kernel_size, kernel_strides, kernel_padding,
          nullptr);

    for (auto& region : state.regions) {
        region->emplaceBlock();
    }

    if (ppe.hasValue()) {
        addPPETask(builder, state, ppe.getValue());
    }
}

void EMU::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output,
                                  mlir::Value input, mlir::Value weights, mlir::Value weightsTable,
                                  VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
                                  mlir::ArrayAttr kernel_strides, mlir::ArrayAttr kernel_padding,
                                  mlir::ArrayAttr rawFilterShape, Optional<VPU::PPETaskAttr> ppe) {
    build(builder, state, output, input, weights, weightsTable,
          VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type), kernel_size, kernel_strides, kernel_padding,
          rawFilterShape);

    for (auto& region : state.regions) {
        region->emplaceBlock();
    }

    if (ppe.hasValue()) {
        addPPETask(builder, state, ppe.getValue());
    }
}

//
// addPPETask
//

// Add PPETaskOps from another a VPU::PPETaskAttr ppeAttr
void EMU::NCEClusterTaskOp::addPPETask(mlir::OpBuilder& builder, mlir::OperationState& state,
                                       VPU::PPETaskAttr ppeAttr) {
    const auto multList =
            ppeAttr.quant_mult() != nullptr
                    ? builder.getI64ArrayAttr(makeArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.quant_mult())))
                    : nullptr;
    const auto shiftList =
            ppeAttr.quant_shift() != nullptr
                    ? builder.getI64ArrayAttr(makeArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.quant_shift())))
                    : nullptr;

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto ppe = state.regions.begin()->get();
    builder.setInsertionPointToEnd(&(ppe->front()));
    builder.create<EMU::PPETaskOp>(state.location, ppeAttr.mode(), ppeAttr.clamp_low(), ppeAttr.clamp_high(),
                                   ppeAttr.lrelu_mult(), ppeAttr.lrelu_shift(), multList, shiftList,
                                   ppeAttr.quant_post_shift());
}

// Copy PPETaskOps from another EMU::NCEClusterTaskOp
void EMU::NCEClusterTaskOp::addPPETask(mlir::OpBuilder& builder, EMU::NCEClusterTaskOp origOp) {
    if (ppe().empty()) {
        ppe().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&ppe().front());

    for (auto ppeOp : origOp.getOps<EMU::PPETaskOp>()) {
        builder.create<EMU::PPETaskOp>(getLoc(), ppeOp.ppe_layer_type(), ppeOp.clamp_lowAttr(), ppeOp.clamp_highAttr(),
                                       ppeOp.lrelu_multAttr(), ppeOp.lrelu_shiftAttr(), ppeOp.quant_multAttr(),
                                       ppeOp.quant_shiftAttr(), ppeOp.quant_post_shiftAttr());
    }
}

//
// verify
//

namespace {

mlir::LogicalResult verifyNCEConv(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::CONV || op.task_type() == VPUIP::NCETaskType::CMCONV,
                      "Expected task type '{0}' or {1}, but got '{2}'", VPUIP::NCETaskType::CONV,
                      VPUIP::NCETaskType::CMCONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.weight_table() != nullptr) {
        const auto weightsShape = getShape(op.weights());
        const auto OC = weightsShape[Dims4D::Filter::OC];

        const auto weightTableShape = getShape(op.weight_table());
        const auto weightTableNumElements = weightTableShape.totalSize();

        if (OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
            return errorAt(op, "Weight table must have '{0}' elements, got '{1}'",
                           OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
        }
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::AVEPOOL || op.task_type() == VPUIP::NCETaskType::MAXPOOL,
                      "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::AVEPOOL,
                      VPUIP::NCETaskType::MAXPOOL, op.task_type());

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEEltwise(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::ELTWISE, op.task_type());

    if (op.weight_table() != nullptr) {
        const auto outputShape = getShape(op.output());
        const auto OC = outputShape[Dims4D::Act::C];

        const auto weightTableShape = getShape(op.weight_table());
        const auto weightTableNumElements = weightTableShape.totalSize();

        if (OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
            return errorAt(op, "Weight table must have '{0}' elements, got '{1}'",
                           OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
        }
    }

    if (op.kernel_sizeAttr() != nullptr) {
        return errorAt(op, "kernel_size should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() != nullptr) {
        return errorAt(op, "kernel_strides should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() != nullptr) {
        return errorAt(op, "kernel_padding should be empty for NCETaskType : '{0}'", op.task_type());
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEDWConv(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::DWCONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.weight_table() != nullptr) {
        const auto weightsShape = getShape(op.weights());
        const auto OC = weightsShape[Dims4D::Filter::OC];

        const auto weightTableShape = getShape(op.weight_table());
        const auto weightTableNumElements = weightTableShape.totalSize();

        if (OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
            return errorAt(op, "Weight table must have '{0}' elements, got '{1}'",
                           OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
        }
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::EMU::NCEClusterTaskOp::verify() {
    const auto op = getOperation();
    if (task_type() == VPUIP::NCETaskType::CONV || task_type() == VPUIP::NCETaskType::CMCONV) {
        if (mlir::failed(verifyNCEConv(*this))) {
            return mlir::failure();
        }
    } else if (task_type() == VPUIP::NCETaskType::MAXPOOL || task_type() == VPUIP::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(*this))) {
            return mlir::failure();
        }
    } else if (task_type() == VPUIP::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(*this))) {
            return mlir::failure();
        }
    } else if (task_type() == VPUIP::NCETaskType::DWCONV) {
        if (mlir::failed(verifyNCEDWConv(*this))) {
            return mlir::failure();
        }
    } else {
        return errorAt(op, "NCE Task Type '{0}' in not supported", task_type());
    }

    for (auto& ppeOp : ppe().getOps()) {
        if (!mlir::isa<EMU::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    return mlir::success();
}

//
// NCEClusterTaskOp::serialize
//

namespace {

MVCNN::DPULayerType getDPULayerType(VPUIP::NCETaskType taskType) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        return MVCNN::DPULayerType_CONV;
    case VPUIP::NCETaskType::DWCONV:
        return MVCNN::DPULayerType_DWCONV;
    case VPUIP::NCETaskType::MAXPOOL:
        return MVCNN::DPULayerType_MAXPOOL;
    case VPUIP::NCETaskType::AVEPOOL:
        return MVCNN::DPULayerType_AVEPOOL;
    case VPUIP::NCETaskType::FCL:
        return MVCNN::DPULayerType_FCL;
    case VPUIP::NCETaskType::ELTWISE:
        return MVCNN::DPULayerType_ELTWISE;
    case VPUIP::NCETaskType::IDENTITY:
        return MVCNN::DPULayerType_IDENTITY;
    case VPUIP::NCETaskType::CMCONV:
        return MVCNN::DPULayerType_CMCONV;
    default:
        VPUX_THROW("Unsupported DPU Layer type: '{0}'", taskType);
    }
}

MVCNN::PPELayerType getPPELayerType(VPU::PPEMode ppeType) {
    switch (ppeType) {
    case VPU::PPEMode::STORE:
        return MVCNN::PPELayerType_STORE;
    case VPU::PPEMode::LOAD:
        return MVCNN::PPELayerType_LOAD;
    case VPU::PPEMode::CLEAR:
        return MVCNN::PPELayerType_CLEAR;
    case VPU::PPEMode::NOOP:
        return MVCNN::PPELayerType_NOOP;
    case VPU::PPEMode::HALT:
        return MVCNN::PPELayerType_HALT;
    case VPU::PPEMode::ADD:
        return MVCNN::PPELayerType_ADD;
    case VPU::PPEMode::SUB:
        return MVCNN::PPELayerType_SUB;
    case VPU::PPEMode::MULT:
        return MVCNN::PPELayerType_MULT;
    case VPU::PPEMode::MAXIMUM:
        return MVCNN::PPELayerType_MAXIMUM;
    case VPU::PPEMode::MINIMUM:
        return MVCNN::PPELayerType_MINIMUM;
    case VPU::PPEMode::AND:
        return MVCNN::PPELayerType_AND;
    case VPU::PPEMode::OR:
        return MVCNN::PPELayerType_OR;
    case VPU::PPEMode::XOR:
        return MVCNN::PPELayerType_XOR;
    case VPU::PPEMode::LRELU:
        return MVCNN::PPELayerType_LRELU;
    case VPU::PPEMode::LRELUX:
        return MVCNN::PPELayerType_LRELUX;
    case VPU::PPEMode::LPRELU:
        return MVCNN::PPELayerType_LPRELU;
    case VPU::PPEMode::CEIL:
        return MVCNN::PPELayerType_CEIL;
    case VPU::PPEMode::FLOOR:
        return MVCNN::PPELayerType_FLOOR;
    case VPU::PPEMode::EXP:
        return MVCNN::PPELayerType_EXP;
    case VPU::PPEMode::SIGMOID:
        return MVCNN::PPELayerType_SIGMOID;
    case VPU::PPEMode::TANH:
        return MVCNN::PPELayerType_TANH;
    case VPU::PPEMode::SQRT:
        return MVCNN::PPELayerType_SQRT;
    case VPU::PPEMode::RSQRT:
        return MVCNN::PPELayerType_RSQRT;
    case VPU::PPEMode::FLEXARB:
        return MVCNN::PPELayerType_FLEXARB;
    case VPU::PPEMode::NOT:
        return MVCNN::PPELayerType_NOT;
    case VPU::PPEMode::ABS:
        return MVCNN::PPELayerType_ABS;
    case VPU::PPEMode::NEG:
        return MVCNN::PPELayerType_NEG;
    default:
        VPUX_THROW("Unsupported PPE Layer type: '{0}'", ppeType);
    }
}

}  // namespace

// This is a helper routine to build new TensorReference out of NCE task output with provided
// quantization scale parameters
EMU::BlobWriter::TensorReference getTensorReferenceWithUpdatedQuantParams(EMU::NCEClusterTaskOp* nceTask,
                                                                          EMU::BlobWriter& writer,
                                                                          ArrayRef<int64_t> ppeQuantMult,
                                                                          ArrayRef<int64_t> ppeQuantShift,
                                                                          int64_t ppeQuantPostShift) {
    // Get also ZP from output
    SmallVector<uint8_t> quantZeroPoints;

    auto outputElementType = nceTask->output().getType().cast<mlir::ShapedType>().getElementType();
    if (const auto uniformQuantType = outputElementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        quantZeroPoints.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType =
                       outputElementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto zp_output = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp_output.size());
        std::transform(zp_output.begin(), zp_output.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    VPUX_THROW_UNLESS(ppeQuantShift.size() == quantZeroPoints.size(),
                      "Mismatch of size between quant shift/mult vector and quant ZP:  {0} != {1}",
                      ppeQuantShift.size(), quantZeroPoints.size());

    ArrayRef<uint8_t> zeroPointsArrRef = makeArrayRef(quantZeroPoints);

    const auto outDataTensorRef = writer.getTensorReference(nceTask->output());
    const auto outDataName = flatbuffers::GetString(outDataTensorRef->name());
    return writer.replaceTensor(nceTask->output(), llvm::formatv("{0}-quant-params", outDataName).str(),
                                nceTask->output().getType().cast<mlir::ShapedType>(), ppeQuantMult, ppeQuantShift,
                                ppeQuantPostShift, zeroPointsArrRef);
}

EMU::BlobWriter::SpecificTask EMU::NCEClusterTaskOp::serialize(EMU::BlobWriter& writer) {
    SmallVector<uint8_t> ppeList;
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t LreluMult = 1;
    uint32_t LreluShift = 0;
    ::llvm::Optional<SmallVector<int64_t>> ppeQuantMult;
    ::llvm::Optional<SmallVector<int64_t>> ppeQuantShift;
    ::llvm::Optional<int64_t> ppeQuantPostShift;

    for (auto ppeOp : ppe().getOps<EMU::PPETaskOp>()) {
        const auto type = getPPELayerType(ppeOp.ppe_layer_type());
        if (type != MVCNN::PPELayerType_NOOP) {
            ppeList.push_back(type);
        }
        if (ppeOp.clamp_low().hasValue()) {
            clampLow = checked_cast<int32_t>(ppeOp.clamp_low().getValue());
        }
        if (ppeOp.clamp_high().hasValue()) {
            clampHigh = checked_cast<int32_t>(ppeOp.clamp_high().getValue());
        }
        if (ppeOp.lrelu_mult().hasValue()) {
            LreluMult = checked_cast<int32_t>(ppeOp.lrelu_mult().getValue());
        }
        if (ppeOp.lrelu_shift().hasValue()) {
            LreluShift = checked_cast<uint32_t>(ppeOp.lrelu_shift().getValue());
        }
        if (ppeOp.quant_mult().hasValue()) {
            ppeQuantMult = parseIntArrayAttr<int64_t>(ppeOp.quant_mult().getValue());
        }
        if (ppeOp.quant_shift().hasValue()) {
            ppeQuantShift = parseIntArrayAttr<int64_t>(ppeOp.quant_shift().getValue());
        }
        if (ppeOp.quant_post_shift().hasValue()) {
            ppeQuantPostShift = checked_cast<int64_t>(ppeOp.quant_post_shift().getValue());
        }
    }
    VPUX_THROW_UNLESS(ppeList.size() <= 1, "Cannot set more than one PPE task");

    auto ppeLayerTypes = writer.createVector(ppeList);
    // TODO: Lrelu_Mult, Lrelu_Shift
    auto ppeFixedFunction =
            MVCNN::CreatePPEFixedFunction(writer, ppeLayerTypes, clampLow, clampHigh, LreluMult, LreluShift);
    // TODO: scale_data, rounding, instruction_list_data
    auto ppeTask = MVCNN::CreatePPETask(writer, 0, ppeFixedFunction);

    int16_t kernelSizeH = 1, kernelSizeW = 1;
    int16_t kernelStridesH = 1, kernelStridesW = 1;
    int16_t kernelPadL = 0, kernelPadR = 0, kernelPadT = 0, kernelPadB = 0;

    if (kernel_sizeAttr() != nullptr) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(kernel_sizeAttr());
        kernelSizeH = checked_cast<int16_t>(kernelSize[0]);
        kernelSizeW = checked_cast<int16_t>(kernelSize[1]);
    }

    if (kernel_stridesAttr() != nullptr) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(kernel_stridesAttr());
        kernelStridesH = checked_cast<int16_t>(kernelStrides[0]);
        kernelStridesW = checked_cast<int16_t>(kernelStrides[1]);
    }

    if (kernel_paddingAttr() != nullptr) {
        const auto kernelPadding = parseIntArrayAttr<int64_t>(kernel_paddingAttr());
        kernelPadL = checked_cast<int16_t>(kernelPadding[0]);
        kernelPadR = checked_cast<int16_t>(kernelPadding[1]);
        kernelPadT = checked_cast<int16_t>(kernelPadding[2]);
        kernelPadB = checked_cast<int16_t>(kernelPadding[3]);
    }

    const auto inputData = writer.getTensor(input());
    const auto weightsData = weights() != nullptr ? writer.getTensor(weights()) : 0;
    const auto weightsTable = weight_table() != nullptr ? writer.getTensor(weight_table()) : 0;

    VPUX_THROW_UNLESS(weight_table() != nullptr, "WeightsTable is required for task type {0}.", task_type());

    const auto outputData = writer.getTensor(output());
    const auto invariant = MVCNN::CreateNCEInvariantFields(writer,
                                                           getDPULayerType(task_type()),  // dpu_task_type
                                                           ppeTask,                       // ppe_task
                                                           MVCNN::MPE_Mode_VECTOR,        // mpe_frequent_mode
                                                           kernelSizeH,                   // kernelH
                                                           kernelSizeW,                   // kernelW
                                                           kernelStridesH,                // kernel_strideH
                                                           kernelStridesW,                // kernel_strideW
                                                           kernelPadL,                    // kernel_padLeft
                                                           kernelPadR,                    // kernel_padRight
                                                           kernelPadT,                    // kernel_padTop
                                                           kernelPadB,                    // kernel_padBottom
                                                           0,                             // parent_input_tensor
                                                           0,                             // parent_output_tensor
                                                           0,                             // parent_weights_tensor
                                                           inputData,                     // input_data
                                                           outputData,                    // output_data
                                                           weightsData,                   // weights_data
                                                           weightsTable,                  // weights_table
                                                           0,                             // activation_window
                                                           0,  // activation_window_channel_length
                                                           0,  // enabled_optimizations
                                                           0,  // odu_offset
                                                           0,  // out_channel_offset
                                                           0,  // is_segmented
                                                           0   // is_continued
    );

    MVCNN::NCE2TaskBuilder builder(writer);
    builder.add_invariant(invariant);

    return {builder.Finish().Union(), MVCNN::SpecificTask_NCE2Task};
}

namespace {

// SqueezeDWConvWeights

class SqueezeDWConvWeights final : public mlir::OpRewritePattern<EMU::NCEClusterTaskOp> {
public:
    using mlir::OpRewritePattern<EMU::NCEClusterTaskOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(EMU::NCEClusterTaskOp origOp, mlir::PatternRewriter& rewriter) const final;
};

/*
    Emulator impl of DWConv uses 3D weights of shape Channels x KernelH x KernelW.
    Therefore, in case of 4D weights with input channel dim = 1, we squeeze them to get rid of that dim.
*/
mlir::LogicalResult SqueezeDWConvWeights::matchAndRewrite(EMU::NCEClusterTaskOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (origOp.task_type() != VPUIP::NCETaskType::DWCONV) {
        return mlir::failure();
    }

    const auto ctx = rewriter.getContext();
    const auto loc = origOp.getLoc();
    const auto weights = origOp.weights();
    const auto weightsType = weights.getType().cast<vpux::NDTypeInterface>();
    const auto weightsShape = weightsType.getShape();
    const auto filtersPerInChan = weightsShape[Dims4D::Filter::IC];
    if (weightsType.getRank() != 4 || filtersPerInChan != 1) {
        return mlir::failure();
    }

    const auto OC = weightsShape[Dims4D::Filter::OC];
    const auto KY = weightsShape[Dims4D::Filter::KY];
    const auto KX = weightsShape[Dims4D::Filter::KX];

    const auto newWeightsShape = Shape({OC * filtersPerInChan, KY, KX});
    auto newWeightsType =
            vpux::getTensorType(newWeightsShape, weightsType.getElementType(),
                                DimsOrder::fromNumDims(newWeightsShape.size()), weightsType.getMemSpace());

    if (auto weightsConst = weights.getDefiningOp<Const::DeclareOp>()) {
        const auto planarConstAttr = weightsConst.contentAttr().reorder(DimsOrder::OIYX);
        const auto constAttr3d = planarConstAttr.reshape(newWeightsShape);
        auto squeezedWeightsOp = rewriter.create<Const::DeclareOp>(loc, newWeightsType, constAttr3d);

        rewriter.replaceOp(weightsConst, squeezedWeightsOp.output());

        return mlir::success();
    }

    auto weightsOIYX = weights;
    if (weightsType.getDimsOrder() != DimsOrder::OIYX) {
        auto weightsOIYXType = weightsType.changeDimsOrder(DimsOrder::OIYX);
        const auto oiyxAttr = mlir::AffineMapAttr::get(DimsOrder::OIYX.toAffineMap(ctx));
        const auto weightsOIYXDimsAttr =
                mlir::AffineMapAttr::get(getPermutationFromOrders(weightsType.getDimsOrder(), DimsOrder::OIYX, ctx));
        weightsOIYX = rewriter.create<IE::PermuteCastOp>(loc, weightsOIYXType, weights, oiyxAttr, weightsOIYXDimsAttr)
                              .output();
    }

    auto squeezedWeightsOp = rewriter.create<IE::ReshapeOp>(loc, newWeightsType, weightsOIYX, nullptr, false,
                                                            getIntArrayAttr(ctx, newWeightsShape));

    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(
            loc, origOp.getType(), origOp.input(), squeezedWeightsOp.output(), origOp.weight_table(),
            origOp.task_type(), origOp.kernel_sizeAttr(), origOp.kernel_stridesAttr(), origOp.kernel_paddingAttr(),
            origOp.rawFilterShapeAttr());

    nceOp.addPPETask(rewriter, origOp);

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::EMU::NCEClusterTaskOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* ctx) {
    patterns.add<SqueezeDWConvWeights>(ctx);
}
