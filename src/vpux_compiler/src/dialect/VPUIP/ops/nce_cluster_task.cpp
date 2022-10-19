//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// NCEClusterTaskOp::build
//

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output,
                                          mlir::Type output_sparsity_map, mlir::Type profiling_output,
                                          mlir::ValueRange operands, ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() >= 4u && "mismatched number of parameters");
    state.addOperands(operands);

    // Compute value for result_segment_sizes attribute and add it to the attributes dictionary
    auto resultSegmentSizesAttr =
            builder.getI32VectorAttr({1, (output_sparsity_map ? 1 : 0), (profiling_output ? 1 : 0)});

    bool foundAttr = false;
    auto newAttributes = SmallVector<mlir::NamedAttribute>(attributes.begin(), attributes.end());
    for (auto& attribute : newAttributes) {
        if (attribute.first != result_segment_sizesAttrName(state.name)) {
            continue;
        }
        attribute.second = resultSegmentSizesAttr;
        foundAttr = true;
        break;
    }
    if (!foundAttr) {
        newAttributes.emplace_back(result_segment_sizesAttrName(state.name), resultSegmentSizesAttr);
    }

    state.addAttributes(newAttributes);
    for (unsigned i = 0; i != 2; ++i)
        (void)state.addRegion();
    state.addTypes(output);
    if (output_sparsity_map)
        state.addTypes(output_sparsity_map);
    if (profiling_output)
        state.addTypes(profiling_output);
}

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value weights, mlir::Value weight_table,
                                          mlir::Value instruction_list_table, mlir::Value activation_window,
                                          mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
                                          vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
                                          mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
                                          mlir::IntegerAttr activation_window_channel_length,
                                          mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
                                          mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset) {
    build(builder, state, output_buff.getType(), nullptr, nullptr, input, nullptr, nullptr, weights, nullptr,
          weight_table, instruction_list_table, activation_window, parent_input, nullptr, nullptr, parent_output,
          nullptr, output_buff, nullptr, nullptr, task_type, kernel_size, kernel_strides, kernel_padding,
          activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented, out_channel_offset);
}

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output,
                                          mlir::Value input, mlir::Value weights, mlir::Value weight_table,
                                          mlir::Value instruction_list_table, mlir::Value activation_window,
                                          mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
                                          vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
                                          mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
                                          mlir::IntegerAttr activation_window_channel_length,
                                          mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
                                          mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset) {
    build(builder, state, output, nullptr, nullptr, input, nullptr, nullptr, weights, nullptr, weight_table,
          instruction_list_table, activation_window, parent_input, nullptr, nullptr, parent_output, nullptr,
          output_buff, nullptr, nullptr, task_type, kernel_size, kernel_strides, kernel_padding,
          activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented, out_channel_offset);
}

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value weights, mlir::Value weight_table,
                                          mlir::Value instruction_list_table, mlir::Value activation_window,
                                          mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
                                          mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type,
                                          mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
                                          vpux::VPU::PaddingAttr kernel_padding,
                                          mlir::IntegerAttr activation_window_channel_length,
                                          mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
                                          mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset) {
    build(builder, state, output_buff.getType(), nullptr, profiling_data ? profiling_data.getType() : nullptr, input,
          nullptr, nullptr, weights, nullptr, weight_table, instruction_list_table, activation_window, parent_input,
          nullptr, nullptr, parent_output, nullptr, output_buff, nullptr, profiling_data, task_type, kernel_size,
          kernel_strides, kernel_padding, activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented,
          out_channel_offset);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type profiling_output,
        mlir::Value input, mlir::Value weights, mlir::Value weight_table, mlir::Value instruction_list_table,
        mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
        mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
        mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset) {
    build(builder, state, output, nullptr, profiling_output, input, nullptr, nullptr, weights, nullptr, weight_table,
          instruction_list_table, activation_window, parent_input, nullptr, nullptr, parent_output, nullptr,
          output_buff, nullptr, profiling_data, task_type, kernel_size, kernel_strides, kernel_padding,
          activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented, out_channel_offset);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value activation_window,
        mlir::Value parent_input, mlir::Value parent_input_sparsity_map, mlir::Value parent_input_storage_element_table,
        mlir::Value parent_output, mlir::Value parent_output_sparsity_map, mlir::Value output_buff,
        mlir::Value output_sparsity_map_buff, mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type,
        mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset) {
    build(builder, state, output_buff.getType(),
          output_sparsity_map_buff ? output_sparsity_map_buff.getType() : nullptr,
          profiling_data ? profiling_data.getType() : nullptr, input, input_sparsity_map, input_storage_element_table,
          weights, weights_sparsity_map, weight_table, instruction_list_table, activation_window, parent_input,
          parent_input_sparsity_map, parent_input_storage_element_table, parent_output, parent_output_sparsity_map,
          output_buff, output_sparsity_map_buff, profiling_data, task_type, kernel_size, kernel_strides, kernel_padding,
          activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented, out_channel_offset);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type output_sparsity_map,
        mlir::Type profiling_output, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value activation_window,
        mlir::Value parent_input, mlir::Value parent_input_sparsity_map, mlir::Value parent_input_storage_element_table,
        mlir::Value parent_output, mlir::Value parent_output_sparsity_map, mlir::Value output_buff,
        mlir::Value output_sparsity_map_buff, mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type,
        mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset) {
    state.addOperands(input);
    if (input_sparsity_map)
        state.addOperands(input_sparsity_map);
    if (input_storage_element_table)
        state.addOperands(input_storage_element_table);
    if (weights)
        state.addOperands(weights);
    if (weights_sparsity_map)
        state.addOperands(weights_sparsity_map);
    if (weight_table)
        state.addOperands(weight_table);
    if (instruction_list_table)
        state.addOperands(instruction_list_table);
    if (activation_window)
        state.addOperands(activation_window);
    state.addOperands(parent_input);
    if (parent_input_sparsity_map)
        state.addOperands(parent_input_sparsity_map);
    if (parent_input_storage_element_table)
        state.addOperands(parent_input_storage_element_table);
    state.addOperands(parent_output);
    if (parent_output_sparsity_map)
        state.addOperands(parent_output_sparsity_map);
    state.addOperands(output_buff);
    if (output_sparsity_map_buff)
        state.addOperands(output_sparsity_map_buff);
    if (profiling_data)
        state.addOperands(profiling_data);
    state.addAttribute(
            operand_segment_sizesAttrName(state.name),
            builder.getI32VectorAttr({1, (input_sparsity_map ? 1 : 0), (input_storage_element_table ? 1 : 0),
                                      (weights ? 1 : 0), (weights_sparsity_map ? 1 : 0), (weight_table ? 1 : 0),
                                      (instruction_list_table ? 1 : 0), (activation_window ? 1 : 0), 1,
                                      (parent_input_sparsity_map ? 1 : 0), (parent_input_storage_element_table ? 1 : 0),
                                      1, (parent_output_sparsity_map ? 1 : 0), 1, (output_sparsity_map_buff ? 1 : 0),
                                      (profiling_data ? 1 : 0)}));
    state.addAttribute(result_segment_sizesAttrName(state.name),
                       builder.getI32VectorAttr({1, (output_sparsity_map ? 1 : 0), (profiling_output ? 1 : 0)}));
    state.addAttribute(task_typeAttrName(state.name),
                       vpux::VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type));
    if (kernel_size) {
        state.addAttribute(kernel_sizeAttrName(state.name), kernel_size);
    }
    if (kernel_strides) {
        state.addAttribute(kernel_stridesAttrName(state.name), kernel_strides);
    }
    if (kernel_padding) {
        state.addAttribute(kernel_paddingAttrName(state.name), kernel_padding);
    }
    if (activation_window_channel_length) {
        state.addAttribute(activation_window_channel_lengthAttrName(state.name), activation_window_channel_length);
    }
    if (is_continued) {
        state.addAttribute(is_continuedAttrName(state.name), is_continued);
    }
    if (cm_sp_pattern) {
        state.addAttribute(cm_sp_patternAttrName(state.name), cm_sp_pattern);
    }
    if (is_segmented) {
        state.addAttribute(is_segmentedAttrName(state.name), is_segmented);
    }
    if (out_channel_offset) {
        state.addAttribute(out_channel_offsetAttrName(state.name), out_channel_offset);
    }
    (void)state.addRegion();
    (void)state.addRegion();
    state.addTypes(output);
    if (output_sparsity_map)
        state.addTypes(output_sparsity_map);
    if (profiling_output)
        state.addTypes(profiling_output);

    for (auto& region : state.regions) {
        region->emplaceBlock();
    }
}

//
// NCEClusterTaskOp::addDPUTask
//

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr start,
                                                           mlir::ArrayAttr end, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId) {
    if (variants().empty()) {
        variants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&variants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), start, end, pad, mpeMode, clusterId);
}

//
// NCEClusterTaskOp::getNumVariants
//

int64_t vpux::VPUIP::NCEClusterTaskOp::getNumVariants() {
    return variants().getBlocks().front().getOperations().size();
}

//
// NCEClusterTaskOp::inferLayoutInfo
//

void vpux::VPUIP::NCEClusterTaskOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    llvm::TypeSwitch<mlir::Operation*, void>(origOp)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp convOp) {
                const auto arch = VPU::getArch(convOp);

                const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(
                        arch, convOp.input().getType().cast<vpux::NDTypeInterface>());

                if (info.getInput(0) == DimsOrder::NCHW && canUseCMajor) {
                    info.setInput(0, DimsOrder::NCHW);
                } else {
                    info.setInput(0, DimsOrder::NHWC);
                }

                info.setInput(1, DimsOrder::OYXI);

                info.setOutput(0, DimsOrder::NHWC);
            })
            .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp) {
                info.setInput(0, DimsOrder::NHWC);
                info.setInput(1, DimsOrder::OYXI);
                info.setOutput(0, DimsOrder::NHWC);
            })
            .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::AvgPoolOp>([&](IE::AvgPoolOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::AddOp>([&](IE::AddOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::MultiplyOp>([&](IE::MultiplyOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::SubtractOp>([&](IE::SubtractOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::AndOp>([&](IE::AndOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Default([](mlir::Operation* unknownOp) {
                VPUX_THROW("Operation '{0}' is not supported by the DPU", unknownOp->getName());
            });
}

//
// NCEClusterTaskOp::sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPUIP::NCEClusterTaskOp::sparsitySupport() {
    const auto sparseInputs =
            (task_type() == vpux::VPUIP::NCETaskType::CONV) || (task_type() == vpux::VPUIP::NCETaskType::ELTWISE)
                    ? vpux::VPU::SparsitySupport::SPARSE_INPUTS
                    : vpux::VPU::SparsitySupport::NONE;
    return sparseInputs | vpux::VPU::SparsitySupport::SPARSE_OUTPUTS;
}

//
// NCEClusterTaskOp::inferReturnTypes
//

mlir::LogicalResult vpux::VPUIP::NCEClusterTaskOp::inferReturnTypes(
        mlir::MLIRContext*, llvm::Optional<mlir::Location>, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange ranges, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPUIP::NCEClusterTaskOpAdaptor adaptor(operands, attrs, ranges);
    inferredReturnTypes.push_back(adaptor.output_buff().getType());
    if (adaptor.profiling_data() != nullptr) {
        inferredReturnTypes.push_back(adaptor.profiling_data().getType());
    }
    return mlir::success();
}

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyNCEConv(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::CONV || op.task_type() == VPUIP::NCETaskType::CMCONV,
                      "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::CONV,
                      VPUIP::NCETaskType::CMCONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.task_type() == VPUIP::NCETaskType::CMCONV) {
        if (op.activation_window() == nullptr) {
            return errorAt(op, "activation_window is required for NCETaskType : '{0}'", op.task_type());
        }
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

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.kernel_stridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.kernel_paddingAttr();
    const auto padLeft = kernelPadding.left().getInt();
    const auto padRight = kernelPadding.right().getInt();
    const auto padTop = kernelPadding.top().getInt();
    const auto padBottom = kernelPadding.bottom().getInt();

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft,
                                                       padRight, arch))) {
        return mlir::failure();
    }

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC > weightTableNumElements) {
        return errorAt(op, "Weight table must have elements greater than or equal to '{0}', got '{1}'",
                       OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
    }

    const auto inOrder = DimsOrder::fromValue(op.input());
    const auto weightsOrder = DimsOrder::fromValue(op.weights());
    const auto outOrder = DimsOrder::fromValue(op.output_buff());

    if (op.task_type() == VPUIP::NCETaskType::CONV && inOrder != DimsOrder::NHWC) {
        return errorAt(op, "For NCE z-major convolution input must have NHWC layout, got '{0}'", inOrder);
    }
    if (op.task_type() == VPUIP::NCETaskType::CMCONV && inOrder != DimsOrder::NCHW) {
        return errorAt(op, "For NCE c-major convolution input must have NCHW layout, got '{0}'", inOrder);
    }
    if (weightsOrder != DimsOrder::OYXI) {
        return errorAt(op, "For NCE convolution weights must have OYXI layout, got '{0}'", weightsOrder);
    }
    if (arch != VPU::ArchKind::VPUX37XX && outOrder != DimsOrder::NHWC) {
        return errorAt(op, "For NCE convolution output must have NHWC layout, got '{0}'", outOrder);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::AVEPOOL || op.task_type() == VPUIP::NCETaskType::MAXPOOL,
                      "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::AVEPOOL,
                      VPUIP::NCETaskType::MAXPOOL, op.task_type());

    // VPUX37XX hw doesn't require weights table and activation window for max/average pool ops
    if (arch != VPU::ArchKind::VPUX37XX) {
        if (op.weight_table() == nullptr) {
            return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
        }
        if (op.activation_window() == nullptr) {
            return errorAt(op, "activation_window is required for NCETaskType : '{0}'", op.task_type());
        }
        if (op.activation_window_channel_lengthAttr() == nullptr) {
            return errorAt(op, "activation_window_channel_length is required for NCETaskType : '{0}'", op.task_type());
        }
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

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.kernel_stridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.kernel_paddingAttr();
    const auto padLeft = kernelPadding.left().getInt();
    const auto padRight = kernelPadding.right().getInt();
    const auto padTop = kernelPadding.top().getInt();
    const auto padBottom = kernelPadding.bottom().getInt();

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft,
                                                       padRight, arch))) {
        return mlir::failure();
    }

    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC}).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEEltwise(VPUIP::NCEClusterTaskOp op, VPU::ArchKind) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::ELTWISE, op.task_type());

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

mlir::LogicalResult verifyNCEDWConv(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::DWCONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.activation_window() == nullptr) {
        return errorAt(op, "activation_window is required for NCETaskType : '{0}'", op.task_type());
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

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.kernel_stridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.kernel_paddingAttr();
    const auto padLeft = kernelPadding.left().getInt();
    const auto padRight = kernelPadding.right().getInt();
    const auto padTop = kernelPadding.top().getInt();
    const auto padBottom = kernelPadding.bottom().getInt();

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft,
                                                       padRight, arch))) {
        return mlir::failure();
    }

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC > weightTableNumElements) {
        return errorAt(op, "Weight table must have elements greater than or equal to '{0}' elements, got '{1}'",
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

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(VPUIP::DPUTaskOp op) {
    static const size_t NUM_WORKLOAD_DIMS = 3;

    if (op.start().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.start().size());
    }
    if (op.end().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.end().size());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::verifyOp(VPUIP::NCEClusterTaskOp op) {
    const auto arch = VPU::getArch(op.getOperation()->getParentOfType<mlir::ModuleOp>());

    for (const auto& operand : op.getOpOperands()) {
        const auto val = operand.get();
        const auto type = val.getType().cast<vpux::NDTypeInterface>().getElementType();

        if (arch != VPU::ArchKind::VPUX37XX && type.isBF16()) {
            return errorAt(op, "BF16 is only supported by VPUX37XX");
        }
    }

    if (op.task_type() == VPUIP::NCETaskType::CONV || op.task_type() == VPUIP::NCETaskType::CMCONV) {
        if (mlir::failed(verifyNCEConv(op, arch))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::MAXPOOL || op.task_type() == VPUIP::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(op, arch))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(op, arch))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::DWCONV) {
        if (mlir::failed(verifyNCEDWConv(op, arch))) {
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

    static const size_t MIN_NUM_DPUS_PER_CLUSTER = 1;

    if (numDPUTasks < MIN_NUM_DPUS_PER_CLUSTER) {
        return errorAt(op, "There should be at least {0} DPU Tasks per NCEClusterTask, but got {1}",
                       MIN_NUM_DPUS_PER_CLUSTER, numDPUTasks);
    }

    for (auto& ppeOp : op.ppe().getOps()) {
        if (!mlir::isa<VPUIP::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    const auto appendToVector = [](SmallVector<mlir::Value>& operands, mlir::Value val) {
        if (val != nullptr)
            operands.push_back(val);
    };
    auto nnCMXOperands = SmallVector<mlir::Value>();

    const auto inputShape = getShape(op.input());
    const auto inputBatch = inputShape[Dims4D::Act::N];
    if (inputBatch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
        return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", inputBatch,
                       vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE);
    }
    appendToVector(nnCMXOperands, op.input());
    appendToVector(nnCMXOperands, op.weights());
    appendToVector(nnCMXOperands, op.weight_table());
    appendToVector(nnCMXOperands, op.activation_window());
    appendToVector(nnCMXOperands, op.output_buff());
    appendToVector(nnCMXOperands, op.profiling_data());

    const auto checkMemoryKind = [&op](mlir::ValueRange operands, EnumSet<VPU::MemoryKind> acceptedMemoryKinds) {
        for (const auto& val : operands) {
            const auto type = val.getType().cast<vpux::NDTypeInterface>();

            const auto mem = type.getMemoryKind();
            if (llvm::find(acceptedMemoryKinds, mem) == acceptedMemoryKinds.end())
                return errorAt(op, "Can't operate with '{0}' MemoryKind.", mem);
        }
        return mlir::success();
    };

    const auto nncmxStatus = checkMemoryKind(
            nnCMXOperands, EnumSet<VPU::MemoryKind>({VPU::MemoryKind::CMX_NN, VPU::MemoryKind::Register}));
    if (nncmxStatus.failed())
        return nncmxStatus;

    // TODO revisit memory checks for parent operands

    for (const auto& val : op.getOperands()) {
        const auto type = val.getType().cast<vpux::NDTypeInterface>();
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

MVCNN::MPE_Mode getMPEMode(VPU::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR:
        return MVCNN::MPE_Mode_VECTOR;
    case VPU::MPEMode::MATRIX:
        return MVCNN::MPE_Mode_MATRIX;
    case VPU::MPEMode::VECTOR_FP16:
        return MVCNN::MPE_Mode_VECTOR_FP16;
    case VPU::MPEMode::CUBOID_16x16:
        return MVCNN::MPE_Mode_CUBOID_16x16;
    case VPU::MPEMode::CUBOID_8x16:
        return MVCNN::MPE_Mode_CUBOID_8x16;
    case VPU::MPEMode::CUBOID_4x16:
        return MVCNN::MPE_Mode_CUBOID_4x16;
    case VPU::MPEMode::NOP:
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

MVCNN::Permutation getODUPermutationType(DimsOrder outputDimsOrder) {
    if (outputDimsOrder == vpux::DimsOrder::NHWC) {
        return MVCNN::Permutation_ZXY;
    } else if (outputDimsOrder == DimsOrder::fromCode(0x1432)) {  // NWHC
        return MVCNN::Permutation_ZYX;
    } else if (outputDimsOrder == DimsOrder::fromCode(0x1423)) {  // NWCH
        return MVCNN::Permutation_YZX;
    } else if (outputDimsOrder == DimsOrder::fromCode(0x1243)) {  // NCWH
        return MVCNN::Permutation_YXZ;
    } else if (outputDimsOrder == vpux::DimsOrder::NHCW) {
        return MVCNN::Permutation_XZY;
    } else if (outputDimsOrder == vpux::DimsOrder::NCHW) {
        return MVCNN::Permutation_XYZ;
    } else {
        VPUX_THROW("Can't get ODU permutation by output dimsOrder: '{0}'", outputDimsOrder);
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

VPU::MPEMode getMPEFrequentModeFromDPUTasks(mlir::Region& dpuTaskOps) {
    std::unordered_map<VPU::MPEMode, size_t> umap;
    for (auto dpuTaskOp : dpuTaskOps.getOps<VPUIP::DPUTaskOp>()) {
        ++umap[dpuTaskOp.mpe_mode()];
        if (umap.size() > 1) {
            VPUX_THROW("Non-uniform DPU task MPE modes is not supported yet.");
        }
    }
    return umap.begin()->first;
}

// This is a helper routine to build new TensorReference out of NCE task for input, weights and output with provided
// quantization scale parameters
vpux::VPUIP::BlobWriter::TensorReference getTensorReferenceWithUpdatedQuantParams(VPUIP::BlobWriter& writer,
                                                                                  ArrayRef<int64_t> ppeQuantMult,
                                                                                  ArrayRef<int64_t> ppeQuantShift,
                                                                                  int64_t ppeQuantPostShift,
                                                                                  mlir::Value tensor) {
    // Get also ZP
    SmallVector<uint8_t> quantZeroPoints;

    auto type = tensor.getType().cast<vpux::NDTypeInterface>();

    auto elementType = type.getElementType();
    if (const auto uniformQuantType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        quantZeroPoints.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType = elementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp.size());
        std::transform(zp.begin(), zp.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    VPUX_THROW_UNLESS(ppeQuantShift.size() == quantZeroPoints.size(),
                      "Mismatch of size between quant shift/mult vector and quant ZP:  {0} != {1}",
                      ppeQuantShift.size(), quantZeroPoints.size());

    // Find corresponding DeclareBufferOp to get all the data needed to build new TensorReference
    auto bufferOp = tensor.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufferOp != nullptr, "Unable to find parent DeclareBufferOp to build new TensorReference");

    auto sectionIndex = bufferOp.getNonEmptySectionIndex();

    // TODO: Update below in case of sparsity
    return writer.createTensorRef("tensor_scale_updated", type, bufferOp.section(), sectionIndex, bufferOp.byteOffset(),
                                  ppeQuantMult, ppeQuantShift, ppeQuantPostShift, quantZeroPoints, None, None,
                                  bufferOp.swizzlingKey());
}

// This is a helper routine to build new TensorReference of individual variant with profiling data
vpux::VPUIP::BlobWriter::TensorReference getTensorReferenceForVariantProfiling(VPUIP::NCEClusterTaskOp nceTask,
                                                                               VPUIP::BlobWriter& writer,
                                                                               size_t variantId,
                                                                               uint16_t workloadSize) {
    static size_t tempTensorId = 0;

    auto outputType = nceTask.profiling_data().getType().cast<vpux::NDTypeInterface>();
    // Find corresponding DeclareBufferOp to get all the data needed to build new TensorReference
    auto bufferOp = nceTask.profiling_data().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufferOp != nullptr, "Unable to find parent DeclareBufferOp to build new TensorReference");

    auto sectionIndex = bufferOp.getNonEmptySectionIndex();
    const auto refMeta = llvm::formatv("_{0}_dpu_{1}", tempTensorId, variantId).str();
    tempTensorId++;

    return writer.createTensorRef(refMeta, outputType, bufferOp.section(), sectionIndex,
                                  bufferOp.byteOffset() + workloadSize * variantId);
}

std::pair<int32_t, int32_t> getHwpBaseAndWid(const mlir::Value profiling_data) {
    int32_t workload_id = -1;
    int32_t cmx_local_slice_base = 0;
    auto bufferOp = profiling_data.getDefiningOp<VPURT::DeclareBufferOp>();
    const auto byteOffset = bufferOp.byteOffset();

    // workload_id in hardware is 16 bits wide. It is used to address 16-bytes chunks of memory.
    // In total workload_id can address 1 Mb of memory whereas CMX has 2 Mb.
    // That is when CMX local slice base comes into play. It sets the base of addressable 1 Mb.
    // And byte offset is divided by 16 because this is an addressable minimum in our case.
    if (byteOffset < vpux::VPUIP::HW_MIDDLE_OF_AVAILABLE_CMX_MEMORY_3720) {
        cmx_local_slice_base = 0;
        workload_id = static_cast<int32_t>(byteOffset / 16);
    } else {
        cmx_local_slice_base = vpux::VPUIP::HW_MIDDLE_OF_AVAILABLE_CMX_MEMORY_3720;
        workload_id = static_cast<int32_t>((byteOffset - cmx_local_slice_base) / 16);
    }
    return {cmx_local_slice_base, workload_id};
}

float readReluMult(const int32_t Lrelu_Mult) {
    const int32_t LRELU_MULT_MASK = 0x7FF;
    VPUX_THROW_WHEN(Lrelu_Mult & ~LRELU_MULT_MASK, "ReLU multiplier does not match the expected format");
    return static_cast<float>(Lrelu_Mult);
}

int32_t readReluShift(const int32_t Lrelu_Shift) {
    const int32_t LRELU_SHIFT_MASK = 0x1F;
    VPUX_THROW_WHEN(Lrelu_Shift & ~LRELU_SHIFT_MASK, "ReLU shift does not match the expected format");
    return Lrelu_Shift;
}

float calculateFpPReluAlpha(const VPU::ArchKind& arch, const ArrayRef<uint8_t> ppeList, const int32_t LreluMult,
                            const int32_t LreluShift) {
    if (ppeList.empty()) {
        return 1.f;
    }
    const uint8_t ppeType = ppeList[0];
    // Only applicable for LPRELU mode on VPU37XX and VPUX40XX
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
            VPU::ArchKind::VPUX40XX,
    };
    if (compatibleTargets.count(arch) == 0 || ppeType != MVCNN::PPELayerType_LPRELU) {
        return 1.f;
    }

    const auto lReluMult = readReluMult(LreluMult);
    const auto lReluShift = readReluShift(LreluShift);
    return std::ldexp(lReluMult, -lReluShift);
}

}  // namespace

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NCEClusterTaskOp::serialize(VPUIP::BlobWriter& writer) {
    const auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const bool isSwProfiling = (arch != VPU::ArchKind::VPUX37XX);
    const bool isProfEnabled = profiling_data() != nullptr;

    const uint16_t wlSize = VPUIP::getProfWorkloadSize(module);
    const std::pair<int32_t, int32_t> hwpBaseAndWid = (!isSwProfiling && isProfEnabled)
                                                              ? getHwpBaseAndWid(profiling_data())
                                                              : std::make_pair<int32_t, int32_t>(0, -1);

    SmallVector<flatbuffers::Offset<MVCNN::NCEVariantFields>> variantList;
    size_t profiledDpuTasksCount = 0;
    for (auto dpuTaskOp : variants().getOps<VPUIP::DPUTaskOp>()) {
        const auto start = parseIntArrayAttr<int64_t>(dpuTaskOp.start());
        const auto end = parseIntArrayAttr<int64_t>(dpuTaskOp.end());
        const auto pad = dpuTaskOp.pad();

        flatbuffers::Offset<MVCNN::TensorReference> profilingData = {0};
        int32_t current_workload_id = hwpBaseAndWid.second;
        if (isProfEnabled) {
            if (isSwProfiling) {
                // For software DPU profiling we don't care about workload_id, it may be -1.
                // Calculating here correct tensor reference with a shift for particular variant.
                profilingData = getTensorReferenceForVariantProfiling(*this, writer, profiledDpuTasksCount, wlSize);
            } else {
                // Hardware profiling used. Invariant uses CMX base address and variant needs a workload_id
                // calculated as wid per NCE cluster task + shift per variant.
                current_workload_id += static_cast<int32_t>(profiledDpuTasksCount);
            }
        }

        const auto variant = MVCNN::CreateNCEVariantFields(writer,
                                                           0,                                            // Barriers
                                                           getMPEMode(dpuTaskOp.mpe_mode()),             // MPE mode
                                                           static_cast<int16_t>(pad.left().getInt()),    // padLeft
                                                           static_cast<int16_t>(pad.right().getInt()),   // padRight
                                                           static_cast<int16_t>(pad.top().getInt()),     // padTop
                                                           static_cast<int16_t>(pad.bottom().getInt()),  // padBottom
                                                           static_cast<int16_t>(start[0]),  // workload_start_X
                                                           static_cast<int16_t>(start[1]),  // workload_start_Y
                                                           static_cast<int16_t>(start[2]),  // workload_start_Z
                                                           static_cast<int16_t>(end[0]),    // workload_end_X
                                                           static_cast<int16_t>(end[1]),    // workload_end_Y
                                                           static_cast<int16_t>(end[2]),    // workload_end_Z
                                                           0,                               // flex_map_column
                                                           0,                               // flex_map_array
                                                           0,                               // flex_inner
                                                           0,                               // flex_outer
                                                           0,                               // flex_outer_order
                                                           profilingData,                   // profiling_data
                                                           0,                               // halo_regions
                                                           current_workload_id              // workload_id
        );
        ++profiledDpuTasksCount;
        variantList.push_back(variant);
    }
    const auto variant = writer.createVector(variantList);

    SmallVector<uint8_t> ppeList;
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t LreluMult = 1;
    uint32_t LreluShift = 0;
    ::llvm::Optional<SmallVector<int64_t>> ppeQuantMult;
    ::llvm::Optional<SmallVector<int64_t>> ppeQuantShift;
    ::llvm::Optional<int64_t> ppeQuantPostShift;
    ::llvm::Optional<float> ppeQuantScale;
    ::llvm::Optional<SmallVector<int64_t>> in1QuantMult;
    ::llvm::Optional<SmallVector<int64_t>> in2QuantMult;

    for (auto ppeOp : ppe().getOps<VPUIP::PPETaskOp>()) {
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
        if (ppeOp.quant_scale().hasValue()) {
            auto floatScaleAttr = ppeOp.quant_scaleAttr().getValue()[0];
            ppeQuantScale = static_cast<float>(floatScaleAttr.dyn_cast_or_null<mlir::FloatAttr>().getValueAsDouble());
        }
        if (ppeOp.in1_quant_mult().hasValue()) {
            in1QuantMult = parseIntArrayAttr<int64_t>(ppeOp.in1_quant_mult().getValue());
        }
        if (ppeOp.in2_quant_mult().hasValue()) {
            in2QuantMult = parseIntArrayAttr<int64_t>(ppeOp.in2_quant_mult().getValue());
        }
    }
    VPUX_THROW_UNLESS(ppeList.size() <= 1, "Cannot set more than one PPE task");

    auto ppeLayerTypes = writer.createVector(ppeList);
    // TODO: Lrelu_Mult, Lrelu_Shift
    auto ppeFixedFunction =
            MVCNN::CreatePPEFixedFunction(writer, ppeLayerTypes, clampLow, clampHigh, LreluMult, LreluShift);
    // TODO: scale_data, rounding
    const auto instructionListTable =
            instruction_list_table() != nullptr ? writer.getTensorRef(instruction_list_table()) : 0;

    const float fpPReluAlpha = calculateFpPReluAlpha(arch, ppeList, LreluMult, LreluShift);
    auto ppeTask = MVCNN::CreatePPETask(writer, 0, ppeFixedFunction, MVCNN::PPERoundingMode_RNE, instructionListTable,
                                        ppeQuantScale.getValueOr(1.0), fpPReluAlpha);

    int16_t kernelSizeH = 1, kernelSizeW = 1;
    int16_t kernelStridesH = 1, kernelStridesW = 1;
    int16_t kernelPadL = 0, kernelPadR = 0, kernelPadT = 0, kernelPadB = 0;
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> enabled_optimizations = 0;
    int32_t odu_offset = 0;
    int32_t out_channel_offset = 0;
    bool is_segmented = false;
    bool is_continued = false;
    uint16_t cm_sp_pattern = 0;

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
        const auto kernelPadding = kernel_paddingAttr();
        kernelPadL = checked_cast<int16_t>(kernelPadding.left().getInt());
        kernelPadR = checked_cast<int16_t>(kernelPadding.right().getInt());
        kernelPadT = checked_cast<int16_t>(kernelPadding.top().getInt());
        kernelPadB = checked_cast<int16_t>(kernelPadding.bottom().getInt());
    }

    is_continued = (is_continuedAttr() != nullptr);
    is_segmented = (is_segmentedAttr() != nullptr);

    // Extract output permutation from output layout
    MVCNN::Permutation oduPermutation = getODUPermutationType(DimsOrder::fromValue(output()));

    if (cm_sp_patternAttr() != nullptr) {
        cm_sp_pattern = checked_cast<uint16_t>(cm_sp_patternAttr().getValue().getSExtValue());
    }

    if (out_channel_offsetAttr() != nullptr) {
        out_channel_offset = checked_cast<int32_t>(out_channel_offsetAttr().getValue().getSExtValue());
    }

    auto inputData = writer.getTensorRef(input());
    auto weightsData = weights() != nullptr ? writer.getTensorRef(weights()) : 0;
    const auto weightsTable = weight_table() != nullptr ? writer.getTensorRef(weight_table()) : 0;
    const auto activationWindow = activation_window() != nullptr ? writer.getTensorRef(activation_window()) : 0;
    const auto activationWindowChannelLength = checked_cast<int32_t>(activation_window_channel_length().getValueOr(0));

    auto outputData = writer.getTensorRef(output());

    // If quant scale (mult, shift) settings were provided as part of PPE block then use it to build new
    // output TensorReference. This is required for Eltwise operation which doesn't have weights table
    // and PPE quantization settings (Mult, Shift) need to be provided for NN runtime in output tensor descriptor
    const auto isQuantizationProvided =
            ppeQuantMult.hasValue() && ppeQuantShift.hasValue() && ppeQuantPostShift.hasValue();
    const auto isQuantizationNotProvided =
            !ppeQuantMult.hasValue() && !ppeQuantShift.hasValue() && !ppeQuantPostShift.hasValue();
    const auto isInputQuantizationProvided = in1QuantMult.hasValue() && in2QuantMult.hasValue();
    VPUX_THROW_WHEN(!isQuantizationProvided && !isQuantizationNotProvided, "Missing quantization scale settings.");

    if (isQuantizationProvided) {
        outputData = getTensorReferenceWithUpdatedQuantParams(writer, ppeQuantMult.getValue(), ppeQuantShift.getValue(),
                                                              ppeQuantPostShift.getValue(), this->output_buff());
    }
    if (isInputQuantizationProvided) {
        // Shifts must be set 0 for VPUX37XX and VPUX40XX runtime to be considered, otherwise runtime will ignore inputs
        // MULT.
        inputData = getTensorReferenceWithUpdatedQuantParams(writer, in1QuantMult.getValue(), {0}, 0, this->input());
        weightsData =
                getTensorReferenceWithUpdatedQuantParams(writer, in2QuantMult.getValue(), {0}, 0, this->weights());
    }

    const auto parentInputTensor = writer.getTensorRef(parent_input());
    const auto parentOutputTensor = writer.getTensorRef(parent_output());

    const auto invariantMPEMode = getMPEFrequentModeFromDPUTasks(variants());

    const auto invariant =
            MVCNN::CreateNCEInvariantFields(writer,
                                            getDPULayerType(task_type()),   // dpu_task_type
                                            ppeTask,                        // ppe_task
                                            getMPEMode(invariantMPEMode),   // mpe_frequent_mode
                                            kernelSizeH,                    // kernelH
                                            kernelSizeW,                    // kernelW
                                            kernelStridesH,                 // kernel_strideH
                                            kernelStridesW,                 // kernel_strideW
                                            kernelPadL,                     // kernel_padLeft
                                            kernelPadR,                     // kernel_padRight
                                            kernelPadT,                     // kernel_padTop
                                            kernelPadB,                     // kernel_padBottom
                                            parentInputTensor,              // parent_input_tensor
                                            parentOutputTensor,             // parent_output_tensor
                                            0,                              // parent_weights_tensor
                                            inputData,                      // input_data
                                            outputData,                     // output_data
                                            weightsData,                    // weights_data
                                            weightsTable,                   // weights_table
                                            activationWindow,               // activation_window
                                            activationWindowChannelLength,  // activation_window_channel_length
                                            enabled_optimizations,          // enabled_optimizations
                                            odu_offset,                     // odu_offset
                                            out_channel_offset,             // out_channel_offset
                                            is_segmented,                   // is_segmented
                                            is_continued,                   // is_continued
                                            false,                          // is_superdense
                                            0,                              // segment_height
                                            oduPermutation,                 // odu_permutation
                                            cm_sp_pattern,                  // cm_sp_pattern
                                            hwpBaseAndWid.first);           // cmx_local_slice_base

    MVCNN::NCE2TaskBuilder builder(writer);
    builder.add_variant(variant);
    builder.add_invariant(invariant);

    return {builder.Finish().Union(), MVCNN::SpecificTask_NCE2Task};
}
