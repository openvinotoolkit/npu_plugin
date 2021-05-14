//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

using namespace vpux;

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value filter, mlir::Value weight_table, mlir::Value activation_window,
                                          mlir::Value parent_input, mlir::Value parent_output, mlir::Value output,
                                          vpux::VPUIP::NCETaskType task_type,
                                          vpux::VPUIP::PPELayerTypeAttr fixed_ppe_task, mlir::ArrayAttr kernel_padding,
                                          mlir::ArrayAttr strides, mlir::ArrayAttr kernel_size) {
    build(builder, state, output.getType(), input, filter, weight_table, activation_window, parent_input, parent_output,
          output, mlir::ValueRange{}, mlir::ValueRange{}, task_type, fixed_ppe_task, kernel_padding, strides,
          kernel_size, 0);
}

vpux::VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr start,
                                                                 mlir::ArrayAttr end, mlir::ArrayAttr padsBegin,
                                                                 mlir::ArrayAttr padsEnd, VPUIP::MPEMode mpeMode) {
    if (variants().empty()) {
        variants().emplaceBlock();
    }
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&variants().back());
    return builder.create<VPUIP::DPUTaskOp>(variants().getLoc(), start, end, padsBegin, padsEnd, mpeMode);
}

namespace {

mlir::LogicalResult checkNCEKernel(VPUIP::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.strides().getValue(), "CheckNCEKernel requires strides");

    const auto strideShape = parseIntArrayAttr(op.strides().getValue());
    const auto kernelSize = parseIntArrayAttr(op.kernel_size().getValue());

    static const int32_t NCE_MAX_KERNEL_SIZE = 11;
    static const int32_t NCE_MAX_STRIDE_SIZE = 8;

    if (kernelSize[0] > NCE_MAX_KERNEL_SIZE || kernelSize[0] <= 0) {
        return errorAt(op, "Unsupported kernel height dimension: '{0}'. Must be between 1-11.", kernelSize[0]);
    }
    if (kernelSize[1] > NCE_MAX_KERNEL_SIZE || kernelSize[1] <= 0) {
        return errorAt(op, "Unsupported kernel height dimension: '{0}'. Must be between 1-11.", kernelSize[1]);
    }

    if (strideShape[0] > NCE_MAX_STRIDE_SIZE || strideShape[0] <= 0) {
        return errorAt(op, "Unsupported stride height dimension: '{0}'. Must be between 1-8.", strideShape[0]);
    }
    if (strideShape[1] > NCE_MAX_STRIDE_SIZE || strideShape[1] <= 0) {
        return errorAt(op, "Unsupported stride width dimension: '{0}'. Must be between 1-8.", strideShape[1]);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEConv(VPUIP::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::CONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.task_type());

    if (!op.filter()) {
        return errorAt(op, "filter is required for NCETaskType : '{0}'", VPUIP::NCETaskType::CONV);
    }
    if (!op.weight_table()) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", VPUIP::NCETaskType::CONV);
    }
    if (!op.strides().hasValue()) {
        return errorAt(op, "strides is required for NCETaskType : '{0}'", VPUIP::NCETaskType::CONV);
    }
    if (!op.kernel_padding().hasValue()) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", VPUIP::NCETaskType::CONV);
    }

    auto p = checkNCEKernel(op);
    if (mlir::failed(p)) {
        return p;
    }

    const auto filterShape = getShape(op.filter());
    const size_t outputChannels = filterShape[Dim(0)];

    static const size_t numElementsPerChannelInWeightTable = 4;
    const auto weightTableShape = getShape(op.weight_table());
    auto weightTableElemTypeSize = getElemTypeSize(op.weight_table().getType());
    auto weightTableNumElements = weightTableShape.totalSize();

    if (outputChannels * numElementsPerChannelInWeightTable != static_cast<size_t>(weightTableShape.totalSize())) {
        return errorAt(op,
                       "Weight table size needs to be {1}*{0} bytes if the number of input channels is {2}, but size "
                       "received is: {2}*{0} bytes",
                       weightTableElemTypeSize, outputChannels * numElementsPerChannelInWeightTable,
                       weightTableNumElements);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEEltwise(VPUIP::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::ELTWISE, op.task_type());

    if (op.filter()) {
        return errorAt(op, "filter should be empty for NCETaskType : '{0}'", VPUIP::NCETaskType::ELTWISE);
    }
    if (op.weight_table()) {
        return errorAt(op, "weight_table should be empty for NCETaskType : '{0}'", VPUIP::NCETaskType::ELTWISE);
    }
    if (op.strides().hasValue()) {
        return errorAt(op, "strides should be empty for NCETaskType : '{0}'", VPUIP::NCETaskType::ELTWISE);
    }
    if (op.kernel_padding().hasValue()) {
        return errorAt(op, "kernel_padding should be empty for NCETaskType : '{0}'", VPUIP::NCETaskType::ELTWISE);
    }
    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(VPUIP::NCEClusterTaskOp op, VPUIP::NCETaskType taskType) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::AVEPOOL || op.task_type() == VPUIP::NCETaskType::MAXPOOL,
                      "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::AVEPOOL,
                      VPUIP::NCETaskType::MAXPOOL, op.task_type());
    if (!op.strides().hasValue()) {
        return errorAt(op, "strides is required for NCETaskType : '{0}'", taskType);
    }
    if (!op.kernel_padding().hasValue()) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", taskType);
    }

    auto p = checkNCEKernel(op);
    if (mlir::failed(p)) {
        return p;
    }

    return mlir::success();
}

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

mlir::LogicalResult vpux::VPUIP::verifyOp(VPUIP::NCEClusterTaskOp op) {
    if (op.task_type() == VPUIP::NCETaskType::CONV) {
        auto check = verifyNCEConv(op);
        if (mlir::failed(check)) {
            return check;
        }
    } else if (op.task_type() == VPUIP::NCETaskType::ELTWISE) {
        auto check = verifyNCEEltwise(op);
        if (mlir::failed(check)) {
            return check;
        }
    } else if (op.task_type() == VPUIP::NCETaskType::AVEPOOL || op.task_type() == VPUIP::NCETaskType::MAXPOOL) {
        auto check = verifyNCEPool(op, op.task_type());
        if (mlir::failed(check)) {
            return check;
        }
    } else {
        VPUX_THROW("NCE Task Type '{0}' support needs to be implemented", op.task_type());
    }

    for (auto& resOp : op.variants().getOps()) {
        if (!mlir::isa<VPUIP::DPUTaskOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'variants' region", resOp.getName(),
                           resOp.getLoc());
        }
    }

    size_t numPPERegions = 0;
    for (auto& ppeRegion : op.ppe_tasks()) {
        if (op.fixed_ppe_task()) {
            return errorAt(op, "'fixed_ppe_task' attribute and 'PPE' region are mutually exclusive, but found both");
        }
        for (auto& resOp : ppeRegion.getOps()) {
            if (!mlir::isa<VPUIP::PPETaskOp>(&resOp)) {
                return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'PPE' region", resOp.getName(),
                               resOp.getLoc());
            }
        }
        ++numPPERegions;
    }
    static const size_t NUM_PPE_REGIONS_PER_CLUSTER = 1;
    if (numPPERegions > NUM_PPE_REGIONS_PER_CLUSTER) {
        return errorAt(op, "There should be at most {0} PPE Region per NCEClusterTask, but got {1}",
                       NUM_PPE_REGIONS_PER_CLUSTER, numPPERegions);
    }

    // TODO: Check that `ppe_tasks` region has a block with valid PPE instructions.

    size_t numDPUTasks = 0;
    for (auto dpuTaskOp : op.variants().getOps<VPUIP::DPUTaskOp>()) {
        static const size_t NUM_WORKLOAD_DIMS = 3;
        if (dpuTaskOp.start().size() != NUM_WORKLOAD_DIMS) {
            return errorAt(op, "Start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, dpuTaskOp.start().size());
        }
        if (dpuTaskOp.end().size() != NUM_WORKLOAD_DIMS) {
            return errorAt(op, "End coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, dpuTaskOp.end().size());
        }
        ++numDPUTasks;
    }
    static const size_t MAX_NUM_DPUS_PER_CLUSTER = 5;
    static const size_t MIN_NUM_DPUS_PER_CLUSTER = 1;
    if (numDPUTasks > MAX_NUM_DPUS_PER_CLUSTER || numDPUTasks < MIN_NUM_DPUS_PER_CLUSTER) {
        return errorAt(op, "There should be a total of {0}-{1} DPU Tasks per NCEClusterTask, but got {2}",
                       MIN_NUM_DPUS_PER_CLUSTER, MAX_NUM_DPUS_PER_CLUSTER, numDPUTasks);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NCEClusterTaskOp::serialize(VPUIP::BlobWriter& writer) {
    SmallVector<flatbuffers::Offset<MVCNN::NCEVariantFields>> nceVariantFieldsList;
    for (auto dpuTaskOp : this->variants().getOps<VPUIP::DPUTaskOp>()) {
        auto padsBegin = parseIntArrayAttr(dpuTaskOp.pads_begin());
        auto padsEnd = parseIntArrayAttr(dpuTaskOp.pads_end());
        auto start = parseIntArrayAttr(dpuTaskOp.start());
        auto end = parseIntArrayAttr(dpuTaskOp.end());

        auto nceVariantFields = MVCNN::CreateNCEVariantFields(writer,
                                                              0,                                   // Barriers
                                                              getMPEMode(dpuTaskOp.mpe_mode()),    // MPE mode
                                                              static_cast<int16_t>(padsBegin[0]),  // padLeft
                                                              static_cast<int16_t>(padsEnd[0]),    // padRight
                                                              static_cast<int16_t>(padsBegin[1]),  // padTop
                                                              static_cast<int16_t>(padsEnd[1]),    // padBottom
                                                              static_cast<int16_t>(start[0]),      // workload_start_X
                                                              static_cast<int16_t>(start[1]),      // workload_start_Y
                                                              static_cast<int16_t>(start[2]),      // workload_start_Z
                                                              static_cast<int16_t>(end[0]),        // workload_end_X
                                                              static_cast<int16_t>(end[1]),        // workload_end_Y
                                                              static_cast<int16_t>(end[2])         // workload_end_Z
        );
        nceVariantFieldsList.push_back(nceVariantFields);
    }
    auto nceVariantFields = writer.createVector(nceVariantFieldsList);

    SmallVector<uint8_t> ppeList;
    if (this->fixed_ppe_task()) {
        auto ppe_layer_type = getPPELayerType(fixed_ppe_task().getValue());
        if (ppe_layer_type != MVCNN::PPELayerType_NOOP) {
            ppeList.push_back(ppe_layer_type);
        }
    } else {
        // TODO: implement generic PPE serialization
        VPUX_THROW_UNLESS(this->ppe_tasks().empty(), "Generic PPE not yet implemented.");
    }
    auto ppeLayerTypes = writer.createVector(ppeList);
    auto ppeFixedFunction = MVCNN::CreatePPEFixedFunction(writer, ppeLayerTypes);
    auto ppeTask = MVCNN::CreatePPETask(writer, 0, ppeFixedFunction);

    auto kernelHeight = this->kernel_size() ? parseIntArrayAttr(this->kernel_size().getValue())[0] : 0;
    auto kernelWidth = this->kernel_size() ? parseIntArrayAttr(this->kernel_size().getValue())[1] : 0;

    auto kernelStridesHeight = this->strides() ? parseIntArrayAttr(this->strides().getValue())[0] : 0;
    auto kernelStridesWidth = this->strides() ? parseIntArrayAttr(this->strides().getValue())[1] : 0;

    auto kernelPaddingLeft = this->kernel_padding() ? parseIntArrayAttr(this->kernel_padding().getValue())[0] : 0;
    auto kernelPaddingRight = this->kernel_padding() ? parseIntArrayAttr(this->kernel_padding().getValue())[1] : 0;
    auto kernelPaddingTop = this->kernel_padding() ? parseIntArrayAttr(this->kernel_padding().getValue())[2] : 0;
    auto kernelPaddingBottom = this->kernel_padding() ? parseIntArrayAttr(this->kernel_padding().getValue())[3] : 0;

    auto parentInputTensor = writer.getTensor(this->parent_input());
    auto parentOutputTensor = writer.getTensor(this->parent_output());
    auto parentWeightsTensor = 0;  // TODO: Finish.

    auto inputData = writer.getTensor(this->input());
    auto outputData = writer.getTensor(this->output());
    auto weightsData = this->filter() ? writer.getTensor(this->filter()) : 0;
    auto weightsTable = this->weight_table() ? writer.getTensor(this->weight_table()) : 0;
    auto activationWindow = this->activation_window() ? writer.getTensor(this->activation_window()) : 0;
    auto activationWindowChannelLength = 4;

    auto invariantMPEMode = getMPEFrequentModeFromDPUTasks(this->variants());

    auto nceInvariantFields =
            MVCNN::CreateNCEInvariantFields(writer,
                                            getDPULayerType(this->task_type()),         // dpu_task_type
                                            ppeTask,                                    // ppe_task
                                            getMPEMode(invariantMPEMode),               // mpe_frequent_mode
                                            static_cast<int16_t>(kernelHeight),         // kernelH
                                            static_cast<int16_t>(kernelWidth),          // kernelW
                                            static_cast<int16_t>(kernelStridesHeight),  // kernel_strideH
                                            static_cast<int16_t>(kernelStridesWidth),   // kernel_strideW
                                            static_cast<int16_t>(kernelPaddingLeft),    // kernel_padLeft
                                            static_cast<int16_t>(kernelPaddingRight),   // kernel_padRight
                                            static_cast<int16_t>(kernelPaddingTop),     // kernel_padTop
                                            static_cast<int16_t>(kernelPaddingBottom),  // kernel_padBottom
                                            parentInputTensor,                          // parent_input_tensor
                                            parentOutputTensor,                         // parent_output_tensor
                                            parentWeightsTensor,                        // parent_weights_tensor
                                            inputData,                                  // input_data
                                            outputData,                                 // output_data
                                            weightsData,                                // weights_data
                                            weightsTable,                               // weights_table
                                            activationWindow,                           // activation_window
                                            activationWindowChannelLength               // activation_window_channel_length
            );

    MVCNN::NCE2TaskBuilder builder(writer);
    builder.add_variant(nceVariantFields);
    builder.add_invariant(nceInvariantFields);

    return {builder.Finish().Union(), MVCNN::SpecificTask_NCE2Task};
}
