//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/utils/extentions.hpp>

using namespace vpux;

//
// NCEClusterTaskOp::build
//

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value weights, mlir::Value weight_table, mlir::Value activation_window,
                                          mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
                                          vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
                                          mlir::ArrayAttr kernel_strides, mlir::ArrayAttr kernel_padding,
                                          mlir::IntegerAttr activation_window_channel_length) {
    build(builder, state, output_buff.getType(), input, weights, weight_table, activation_window, parent_input,
          parent_output, output_buff, mlir::ValueRange{}, mlir::ValueRange{},
          vpux::VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type), kernel_size, kernel_strides,
          kernel_padding, activation_window_channel_length);

    for (auto& region : state.regions) {
        region->emplaceBlock();
    }
}

//
// NCEClusterTaskOp::addDPUTask
//

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr start,
                                                           mlir::ArrayAttr end, mlir::ArrayAttr padsBegin,
                                                           mlir::ArrayAttr padsEnd, VPUIP::MPEMode mpeMode) {
    if (variants().empty()) {
        variants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&variants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), start, end, padsBegin, padsEnd, mpeMode);
}

//
// NCEClusterTaskOp::addPPETask
//

VPUIP::PPETaskOp vpux::VPUIP::NCEClusterTaskOp::addPPETask(mlir::OpBuilder& builder,
                                                           vpux::VPUIP::PPELayerType ppe_layer_type) {
    if (ppe().empty()) {
        ppe().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&ppe().front());

    return builder.create<VPUIP::PPETaskOp>(getLoc(), ppe_layer_type);
}

//
// NCEClusterTaskOp::isSupportedLayout
//

mlir::LogicalResult vpux::VPUIP::NCEClusterTaskOp::isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<IERT::MaxPoolOp>([&](mlir::Operation* op) {
                return isSupportedLayoutSameInOutSpecificDimsOrder(op, info, {DimsOrder::NHWC});
            })
            .Case<IERT::ConvolutionOp>([&](IERT::ConvolutionOp originOp) {
                if (isSupportedLayoutSameInOutSpecificDimsOrder(originOp, info, {DimsOrder::NHWC}).failed()) {
                    // weights layout
                    info.setInput(1, DimsOrder::NHWC);
                    return mlir::failure();
                }

                // check weights layout
                if (!info.hasInput(1) || info.getInput(1) != DimsOrder::NHWC) {
                    fillDataInfo(info, 2, 1, DimsOrder::NHWC);
                    return mlir::failure();
                }

                return mlir::success();
            })
            .Default([](mlir::Operation* unknownOp) -> mlir::LogicalResult {
                VPUX_THROW("Operation '{0}' the operation is not supported by the DPU", unknownOp->getName());
            });
}

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyNCEConv(VPUIP::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::CONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op.kernel_sizeAttr(), op.kernel_stridesAttr(),
                                                       op.kernel_paddingAttr()))) {
        return mlir::failure();
    }

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[IERT::ConvolutionOp::filter_out_channel_dim()];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
        return errorAt(op, "Weight table must have '{0}' elements, got '{1}'",
                       OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
    }

    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC}).failed()) {
        return mlir::failure();
    }

    const auto weightsLayout = DimsOrder::fromValue(op.weights());
    if (weightsLayout != DimsOrder::NHWC) {
        return errorAt(op, "weights layout must be NHWC, got {0}", weightsLayout);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(VPUIP::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::AVEPOOL || op.task_type() == VPUIP::NCETaskType::MAXPOOL,
                      "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::AVEPOOL,
                      VPUIP::NCETaskType::MAXPOOL, op.task_type());

    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.activation_window() == nullptr) {
        return errorAt(op, "activation_window is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.activation_window_channel_lengthAttr() == nullptr) {
        return errorAt(op, "activation_window_channel_length is required for NCETaskType : '{0}'", op.task_type());
    }

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op.kernel_sizeAttr(), op.kernel_stridesAttr(),
                                                       op.kernel_paddingAttr()))) {
        return mlir::failure();
    }

    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC}).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEEltwise(VPUIP::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::ELTWISE, op.task_type());

    if (op.weights() != nullptr) {
        return errorAt(op, "weights should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() != nullptr) {
        return errorAt(op, "weight_table should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.activation_window() != nullptr) {
        return errorAt(op, "activation_window should be empty for NCETaskType : '{0}'", op.task_type());
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

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(VPUIP::DPUTaskOp op) {
    static const size_t NUM_WORKLOAD_DIMS = 3;
    static const size_t NUM_WORKLOAD_PADS = 2;

    if (op.start().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.start().size());
    }
    if (op.end().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.end().size());
    }
    if (op.pads_begin().size() != NUM_WORKLOAD_PADS) {
        return errorAt(op, "pads_begin coords should {0}-D, but got {1}-D", NUM_WORKLOAD_PADS, op.pads_begin().size());
    }
    if (op.pads_end().size() != NUM_WORKLOAD_PADS) {
        return errorAt(op, "pads_end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_PADS, op.pads_end().size());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::verifyOp(VPUIP::NCEClusterTaskOp op) {
    if (op.task_type() == VPUIP::NCETaskType::CONV) {
        if (mlir::failed(verifyNCEConv(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::MAXPOOL || op.task_type() == VPUIP::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(op))) {
            return mlir::failure();
        }
    } else {
        return errorAt(op, "NCE Task Type '{0}' in not supported", op.task_type());
    }

    size_t numDPUTasks = 0;
    for (auto& dpuOp : op.variants().getOps()) {
        if (!mlir::isa<VPUIP::DPUTaskOp>(dpuOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'variants' region", dpuOp.getName());
        }

        ++numDPUTasks;
    }

    static const size_t MAX_NUM_DPUS_PER_CLUSTER = 5;
    static const size_t MIN_NUM_DPUS_PER_CLUSTER = 1;

    if (numDPUTasks > MAX_NUM_DPUS_PER_CLUSTER || numDPUTasks < MIN_NUM_DPUS_PER_CLUSTER) {
        return errorAt(op, "There should be a total of {0}-{1} DPU Tasks per NCEClusterTask, but got {2}",
                       MIN_NUM_DPUS_PER_CLUSTER, MAX_NUM_DPUS_PER_CLUSTER, numDPUTasks);
    }

    for (auto& ppeOp : op.ppe().getOps()) {
        if (!mlir::isa<VPUIP::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    for (const auto& operand : op.getOpOperands()) {
        const auto val = operand.get();
        const auto type = val.getType().cast<mlir::MemRefType>();

        auto mem = getPhysicalMemory(type);
        if (mlir::failed(mem)) {
            return errorAt(op, "Unsupported memory space '{0}'", type.getMemorySpace());
        }
        if (mem.getValue() != PhysicalMemory::CMX_NN) {
            return errorAt(op, "Can't operate with '{0}' PhysicalMemory. Only '{1}' PhysicalMemory is allowed",
                           mem.getValue(), PhysicalMemory::CMX_NN);
        }

        const auto strideReqs = StrideReqs().add(DimStrideReq::compact(MemDim(type.getRank() - 1)));

        if (!strideReqs.checkStrides(val)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", val, strideReqs);
        }
    }

    return mlir::success();
}

//
// NCEClusterTaskOp::serialize
//

namespace {

MVCNN::MPE_Mode getMPEMode(VPUIP::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPUIP::MPEMode::VECTOR:
        return MVCNN::MPE_Mode_VECTOR;
    case VPUIP::MPEMode::MATRIX:
        return MVCNN::MPE_Mode_MATRIX;
    case VPUIP::MPEMode::VECTOR_FP16:
        return MVCNN::MPE_Mode_VECTOR_FP16;
    case VPUIP::MPEMode::CUBOID_16x16:
        return MVCNN::MPE_Mode_CUBOID_16x16;
    case VPUIP::MPEMode::CUBOID_8x16:
        return MVCNN::MPE_Mode_CUBOID_8x16;
    case VPUIP::MPEMode::NOP:
        return MVCNN::MPE_Mode_NOP;
    default:
        VPUX_THROW("Unsupported MPE mode type: '{0}'", mpeMode);
    }
}

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

MVCNN::PPELayerType getPPELayerType(VPUIP::PPELayerType ppeType) {
    switch (ppeType) {
    case VPUIP::PPELayerType::STORE:
        return MVCNN::PPELayerType_STORE;
    case VPUIP::PPELayerType::LOAD:
        return MVCNN::PPELayerType_LOAD;
    case VPUIP::PPELayerType::CLEAR:
        return MVCNN::PPELayerType_CLEAR;
    case VPUIP::PPELayerType::NOOP:
        return MVCNN::PPELayerType_NOOP;
    case VPUIP::PPELayerType::HALT:
        return MVCNN::PPELayerType_HALT;
    case VPUIP::PPELayerType::ADD:
        return MVCNN::PPELayerType_ADD;
    case VPUIP::PPELayerType::SUB:
        return MVCNN::PPELayerType_SUB;
    case VPUIP::PPELayerType::MULT:
        return MVCNN::PPELayerType_MULT;
    case VPUIP::PPELayerType::MAXIMUM:
        return MVCNN::PPELayerType_MAXIMUM;
    case VPUIP::PPELayerType::MINIMUM:
        return MVCNN::PPELayerType_MINIMUM;
    case VPUIP::PPELayerType::AND:
        return MVCNN::PPELayerType_AND;
    case VPUIP::PPELayerType::OR:
        return MVCNN::PPELayerType_OR;
    case VPUIP::PPELayerType::XOR:
        return MVCNN::PPELayerType_XOR;
    case VPUIP::PPELayerType::LRELU:
        return MVCNN::PPELayerType_LRELU;
    case VPUIP::PPELayerType::LRELUX:
        return MVCNN::PPELayerType_LRELUX;
    case VPUIP::PPELayerType::LPRELU:
        return MVCNN::PPELayerType_LPRELU;
    case VPUIP::PPELayerType::CEIL:
        return MVCNN::PPELayerType_CEIL;
    case VPUIP::PPELayerType::FLOOR:
        return MVCNN::PPELayerType_FLOOR;
    case VPUIP::PPELayerType::EXP:
        return MVCNN::PPELayerType_EXP;
    case VPUIP::PPELayerType::SIGMOID:
        return MVCNN::PPELayerType_SIGMOID;
    case VPUIP::PPELayerType::TANH:
        return MVCNN::PPELayerType_TANH;
    case VPUIP::PPELayerType::SQRT:
        return MVCNN::PPELayerType_SQRT;
    case VPUIP::PPELayerType::RSQRT:
        return MVCNN::PPELayerType_RSQRT;
    case VPUIP::PPELayerType::FLEXARB:
        return MVCNN::PPELayerType_FLEXARB;
    case VPUIP::PPELayerType::NOT:
        return MVCNN::PPELayerType_NOT;
    case VPUIP::PPELayerType::ABS:
        return MVCNN::PPELayerType_ABS;
    case VPUIP::PPELayerType::NEG:
        return MVCNN::PPELayerType_NEG;
    default:
        VPUX_THROW("Unsupported PPE Layer type: '{0}'", ppeType);
    }
}

VPUIP::MPEMode getMPEFrequentModeFromDPUTasks(mlir::Region& dpuTaskOps) {
    std::unordered_map<VPUIP::MPEMode, size_t> umap;
    for (auto dpuTaskOp : dpuTaskOps.getOps<VPUIP::DPUTaskOp>()) {
        ++umap[dpuTaskOp.mpe_mode()];
        if (umap.size() > 1) {
            VPUX_THROW("Non-uniform DPU task MPE modes is not supported yet.");
        }
    }
    return umap.begin()->first;
}

}  // namespace

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NCEClusterTaskOp::serialize(VPUIP::BlobWriter& writer) {
    SmallVector<flatbuffers::Offset<MVCNN::NCEVariantFields>> variantList;
    for (auto dpuTaskOp : variants().getOps<VPUIP::DPUTaskOp>()) {
        const auto start = parseIntArrayAttr(dpuTaskOp.start());
        const auto end = parseIntArrayAttr(dpuTaskOp.end());
        const auto padsBegin = parseIntArrayAttr(dpuTaskOp.pads_begin());
        const auto padsEnd = parseIntArrayAttr(dpuTaskOp.pads_end());

        // TODO: [Track number: E#13226]
        // Make padding indexing more obvious
        const auto variant = MVCNN::CreateNCEVariantFields(writer,
                                                           0,                                   // Barriers
                                                           getMPEMode(dpuTaskOp.mpe_mode()),    // MPE mode
                                                           static_cast<int16_t>(padsBegin[1]),  // padLeft
                                                           static_cast<int16_t>(padsEnd[1]),    // padRight
                                                           static_cast<int16_t>(padsBegin[0]),  // padTop
                                                           static_cast<int16_t>(padsEnd[0]),    // padBottom
                                                           static_cast<int16_t>(start[0]),      // workload_start_X
                                                           static_cast<int16_t>(start[1]),      // workload_start_Y
                                                           static_cast<int16_t>(start[2]),      // workload_start_Z
                                                           static_cast<int16_t>(end[0]),        // workload_end_X
                                                           static_cast<int16_t>(end[1]),        // workload_end_Y
                                                           static_cast<int16_t>(end[2])         // workload_end_Z
        );
        variantList.push_back(variant);
    }
    const auto variant = writer.createVector(variantList);

    SmallVector<uint8_t> ppeList;
    for (auto ppeOp : ppe().getOps<VPUIP::PPETaskOp>()) {
        ppeList.push_back(getPPELayerType(ppeOp.ppe_layer_type()));
    }
    auto ppeLayerTypes = writer.createVector(ppeList);
    // TODO: Clamp_Low, Clamp_High, Lrelu_Mult, Lrelu_Shift
    auto ppeFixedFunction = MVCNN::CreatePPEFixedFunction(writer, ppeLayerTypes);
    // TODO: scale_data, rounding, instruction_list_data
    auto ppeTask = MVCNN::CreatePPETask(writer, 0, ppeFixedFunction);

    int16_t kernelSizeH = 1, kernelSizeW = 1;
    int16_t kernelStridesH = 1, kernelStridesW = 1;
    int16_t kernelPadL = 0, kernelPadR = 0, kernelPadT = 0, kernelPadB = 0;

    if (kernel_sizeAttr() != nullptr) {
        const auto kernelSize = parseIntArrayAttr(kernel_sizeAttr());
        kernelSizeH = checked_cast<int16_t>(kernelSize[0]);
        kernelSizeW = checked_cast<int16_t>(kernelSize[1]);
    }

    if (kernel_stridesAttr() != nullptr) {
        const auto kernelStrides = parseIntArrayAttr(kernel_stridesAttr());
        kernelStridesH = checked_cast<int16_t>(kernelStrides[0]);
        kernelStridesW = checked_cast<int16_t>(kernelStrides[1]);
    }

    if (kernel_paddingAttr() != nullptr) {
        const auto kernelPadding = parseIntArrayAttr(kernel_paddingAttr());
        kernelPadL = checked_cast<int16_t>(kernelPadding[0]);
        kernelPadR = checked_cast<int16_t>(kernelPadding[1]);
        kernelPadT = checked_cast<int16_t>(kernelPadding[2]);
        kernelPadB = checked_cast<int16_t>(kernelPadding[3]);
    }

    const auto inputData = writer.getTensor(input());
    const auto weightsData = weights() != nullptr ? writer.getTensor(weights()) : 0;
    const auto weightsTable = weight_table() != nullptr ? writer.getTensor(weight_table()) : 0;
    const auto activationWindow = activation_window() != nullptr ? writer.getTensor(activation_window()) : 0;
    const auto activationWindowChannelLength = activation_window_channel_length().getValueOr(0);

    const auto outputData = writer.getTensor(output());

    const auto parentInputTensor = writer.getTensor(parent_input());
    const auto parentOutputTensor = writer.getTensor(parent_output());

    const auto invariantMPEMode = getMPEFrequentModeFromDPUTasks(variants());

    const auto invariant =
            MVCNN::CreateNCEInvariantFields(writer,
                                            getDPULayerType(task_type()),  // dpu_task_type
                                            ppeTask,                       // ppe_task
                                            getMPEMode(invariantMPEMode),  // mpe_frequent_mode
                                            kernelSizeH,                   // kernelH
                                            kernelSizeW,                   // kernelW
                                            kernelStridesH,                // kernel_strideH
                                            kernelStridesW,                // kernel_strideW
                                            kernelPadL,                    // kernel_padLeft
                                            kernelPadR,                    // kernel_padRight
                                            kernelPadT,                    // kernel_padTop
                                            kernelPadB,                    // kernel_padBottom
                                            parentInputTensor,             // parent_input_tensor
                                            parentOutputTensor,            // parent_output_tensor
                                            0,                             // parent_weights_tensor
                                            inputData,                     // input_data
                                            outputData,                    // output_data
                                            weightsData,                   // weights_data
                                            weightsTable,                  // weights_table
                                            activationWindow,              // activation_window
                                            activationWindowChannelLength  // activation_window_channel_length
            );

    MVCNN::NCE2TaskBuilder builder(writer);
    builder.add_variant(variant);
    builder.add_invariant(invariant);

    return {builder.Finish().Union(), MVCNN::SpecificTask_NCE2Task};
}
