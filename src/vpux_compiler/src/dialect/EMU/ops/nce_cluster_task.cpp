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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyNCEConv(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::CONV, "Expected task type '{0}', but got '{1}'",
                      EMU::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
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

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
        return errorAt(op, "Weight table must have '{0}' elements, got '{1}'", OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                       weightTableNumElements);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::AVEPOOL || op.task_type() == EMU::NCETaskType::MAXPOOL,
                      "Expected task type '{0}' or '{1}', but got '{2}'", EMU::NCETaskType::AVEPOOL,
                      EMU::NCETaskType::MAXPOOL, op.task_type());

    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
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

    return mlir::success();
}

mlir::LogicalResult verifyNCEEltwise(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      EMU::NCETaskType::ELTWISE, op.task_type());

    if (op.weight_table() != nullptr) {
        return errorAt(op, "weight_table should be empty for NCETaskType : '{0}'", op.task_type());
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
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::DWCONV, "Expected task type '{0}', but got '{1}'",
                      EMU::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
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

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
        return errorAt(op, "Weight table must have '{0}' elements, got '{1}'", OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                       weightTableNumElements);
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::EMU::verifyOp(EMU::NCEClusterTaskOp op) {
    if (op.task_type() == EMU::NCETaskType::CONV) {
        if (mlir::failed(verifyNCEConv(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == EMU::NCETaskType::MAXPOOL || op.task_type() == EMU::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == EMU::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == EMU::NCETaskType::DWCONV) {
        if (mlir::failed(verifyNCEDWConv(op))) {
            return mlir::failure();
        }
    } else {
        return errorAt(op, "NCE Task Type '{0}' in not supported", op.task_type());
    }

    for (auto& ppeOp : op.ppe().getOps()) {
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

MVCNN::DPULayerType getDPULayerType(EMU::NCETaskType taskType) {
    switch (taskType) {
    case EMU::NCETaskType::CONV:
        return MVCNN::DPULayerType_CONV;
    case EMU::NCETaskType::DWCONV:
        return MVCNN::DPULayerType_DWCONV;
    case EMU::NCETaskType::MAXPOOL:
        return MVCNN::DPULayerType_MAXPOOL;
    case EMU::NCETaskType::AVEPOOL:
        return MVCNN::DPULayerType_AVEPOOL;
    case EMU::NCETaskType::FCL:
        return MVCNN::DPULayerType_FCL;
    case EMU::NCETaskType::ELTWISE:
        return MVCNN::DPULayerType_ELTWISE;
    case EMU::NCETaskType::IDENTITY:
        return MVCNN::DPULayerType_IDENTITY;
    case EMU::NCETaskType::CMCONV:
        return MVCNN::DPULayerType_CMCONV;
    default:
        VPUX_THROW("Unsupported DPU Layer type: '{0}'", taskType);
    }
}

MVCNN::PPELayerType getPPELayerType(EMU::PPELayerType ppeType) {
    switch (ppeType) {
    case EMU::PPELayerType::STORE:
        return MVCNN::PPELayerType_STORE;
    case EMU::PPELayerType::LOAD:
        return MVCNN::PPELayerType_LOAD;
    case EMU::PPELayerType::CLEAR:
        return MVCNN::PPELayerType_CLEAR;
    case EMU::PPELayerType::NOOP:
        return MVCNN::PPELayerType_NOOP;
    case EMU::PPELayerType::HALT:
        return MVCNN::PPELayerType_HALT;
    case EMU::PPELayerType::ADD:
        return MVCNN::PPELayerType_ADD;
    case EMU::PPELayerType::SUB:
        return MVCNN::PPELayerType_SUB;
    case EMU::PPELayerType::MULT:
        return MVCNN::PPELayerType_MULT;
    case EMU::PPELayerType::MAXIMUM:
        return MVCNN::PPELayerType_MAXIMUM;
    case EMU::PPELayerType::MINIMUM:
        return MVCNN::PPELayerType_MINIMUM;
    case EMU::PPELayerType::AND:
        return MVCNN::PPELayerType_AND;
    case EMU::PPELayerType::OR:
        return MVCNN::PPELayerType_OR;
    case EMU::PPELayerType::XOR:
        return MVCNN::PPELayerType_XOR;
    case EMU::PPELayerType::LRELU:
        return MVCNN::PPELayerType_LRELU;
    case EMU::PPELayerType::LRELUX:
        return MVCNN::PPELayerType_LRELUX;
    case EMU::PPELayerType::LPRELU:
        return MVCNN::PPELayerType_LPRELU;
    case EMU::PPELayerType::CEIL:
        return MVCNN::PPELayerType_CEIL;
    case EMU::PPELayerType::FLOOR:
        return MVCNN::PPELayerType_FLOOR;
    case EMU::PPELayerType::EXP:
        return MVCNN::PPELayerType_EXP;
    case EMU::PPELayerType::SIGMOID:
        return MVCNN::PPELayerType_SIGMOID;
    case EMU::PPELayerType::TANH:
        return MVCNN::PPELayerType_TANH;
    case EMU::PPELayerType::SQRT:
        return MVCNN::PPELayerType_SQRT;
    case EMU::PPELayerType::RSQRT:
        return MVCNN::PPELayerType_RSQRT;
    case EMU::PPELayerType::FLEXARB:
        return MVCNN::PPELayerType_FLEXARB;
    case EMU::PPELayerType::NOT:
        return MVCNN::PPELayerType_NOT;
    case EMU::PPELayerType::ABS:
        return MVCNN::PPELayerType_ABS;
    case EMU::PPELayerType::NEG:
        return MVCNN::PPELayerType_NEG;
    default:
        VPUX_THROW("Unsupported PPE Layer type: '{0}'", ppeType);
    }
}

}  // namespace

// This is a helper routine to build new TensorReference out of NCE task output with provided
// quantization scale parameters
vpux::EMU::BlobWriter::TensorReference getTensorReferenceWithUpdatedQuantParams(vpux::EMU::NCEClusterTaskOp* nceTask,
                                                                                EMU::BlobWriter& writer,
                                                                                ArrayRef<uint16_t> ppeQuantMult,
                                                                                ArrayRef<uint8_t> ppeQuantShift,
                                                                                int8_t ppeQuantPostShift) {
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

    return writer.createTensor(llvm::formatv("output_tensor_scale_updated").str(),
                               nceTask->output().getType().cast<mlir::ShapedType>(), ppeQuantMult, ppeQuantShift,
                               ppeQuantPostShift, zeroPointsArrRef);
}

EMU::BlobWriter::SpecificTask vpux::EMU::NCEClusterTaskOp::serialize(EMU::BlobWriter& writer) {
    SmallVector<uint8_t> ppeList;
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t LreluMult = 1;
    uint32_t LreluShift = 0;
    ::llvm::Optional<SmallVector<uint16_t>> ppeQuantMult;
    ::llvm::Optional<SmallVector<uint8_t>> ppeQuantShift;
    ::llvm::Optional<int8_t> ppeQuantPostShift;

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
            ppeQuantMult = parseIntArrayAttr<uint16_t>(ppeOp.quant_mult().getValue());
        }
        if (ppeOp.quant_shift().hasValue()) {
            ppeQuantShift = parseIntArrayAttr<uint8_t>(ppeOp.quant_shift().getValue());
        }
        if (ppeOp.quant_post_shift().hasValue()) {
            ppeQuantPostShift = checked_cast<int8_t>(ppeOp.quant_post_shift().getValue());
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

    auto outputData = writer.getTensor(output());

    // If quant scale (mult, shift) settings were provided as part of PPE block then use it to build new
    // output TensorReference. This is required for Eltwise operation which doesn't have weights table
    // and PPE quantization settings (Mult, Shift) need to be provided for NN runtime in output tensor descriptor
    const auto isQuantizationProvided =
            ppeQuantMult.hasValue() && ppeQuantShift.hasValue() && ppeQuantPostShift.hasValue();
    const auto isQuantizationNotProvided =
            !ppeQuantMult.hasValue() && !ppeQuantShift.hasValue() && !ppeQuantPostShift.hasValue();
    VPUX_THROW_WHEN(!isQuantizationProvided && !isQuantizationNotProvided, "Missing quantization scale settings.");

    if (isQuantizationProvided) {
        outputData = getTensorReferenceWithUpdatedQuantParams(this, writer, ppeQuantMult.getValue(),
                                                              ppeQuantShift.getValue(), ppeQuantPostShift.getValue());
    }

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
