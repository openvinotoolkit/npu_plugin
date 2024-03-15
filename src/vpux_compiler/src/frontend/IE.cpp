//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/frontend/IE.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/passes/align_scales.hpp"
#include "vpux/passes/clean_up_fq.hpp"
#include "vpux/passes/convert_variadic_split_to_strided_slice.hpp"
#include "vpux/passes/fuse_scale_in_previous_weights_fq.hpp"
#include "vpux/passes/fuse_scaleshift.hpp"
#include "vpux/passes/propagate_fq.hpp"
#include "vpux/passes/remove_split_concat.hpp"
#include "vpux/passes/replace_onnx_pattern_to_reorg.hpp"

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/Verifier.h>

#include <openvino/core/node.hpp>

#include <openvino/core/type/element_type.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#include "vpux/passes/convert_MVN6_to_MVN1.hpp"
#include "vpux/passes/fuse_mvn.hpp"

#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/pull_through_reduce.hpp>
#include <transformations/common_optimizations/reduce_reshape_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/op_conversions/convert_gather_upgrade.hpp>
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_maxpool_downgrade.hpp>
#include <transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_pad12_downgrade.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_9.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_softmax_upgrade.hpp>
#include <transformations/op_conversions/convert_topk11_downgrade.hpp>
#include <transformations/op_conversions/detection_output_downgrade.hpp>
#include <transformations/op_conversions/einsum_decomposition.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/utils/utils.hpp>

#include <algorithm>
#include <cstddef>

using namespace vpux;
using namespace IE;

namespace {

const std::string NGRAPH_ACT_SPARSITY_STATS_KEY = "activation_sparsity_statistic";

template <typename ResType>
ResType getSparsityStatsFieldChecked(const std::shared_ptr<ov::Model>& model, const std::string& primaryKey,
                                     const std::string& secondaryKey) {
    VPUX_THROW_UNLESS(model->has_rt_info(NGRAPH_ACT_SPARSITY_STATS_KEY, primaryKey, secondaryKey),
                      "Failed to query '{0}/{1}/{2}' from runtime statistics", NGRAPH_ACT_SPARSITY_STATS_KEY,
                      primaryKey, secondaryKey);
    return model->get_rt_info<ResType>(NGRAPH_ACT_SPARSITY_STATS_KEY, primaryKey, secondaryKey);
}
}  // namespace

NGraphImporter::Callback NGraphImporter::getParser(const std::shared_ptr<ov::Node>& op) {
    using DispatchMap = std::map<ov::NodeTypeInfo, Callback>;

#define MAP_ENTRY(_NodeType_) \
    { _NodeType_::get_type_info_static(), &NGraphImporter::parseDispatch<_NodeType_> }

    static const DispatchMap dispatchMap{
            {ov::op::v0::Parameter::get_type_info_static(), &NGraphImporter::parseEmpty},
            {ov::op::v0::Result::get_type_info_static(), &NGraphImporter::parseEmpty},

            MAP_ENTRY(opset_latest::Constant),
            MAP_ENTRY(opset_latest::Convert),
            MAP_ENTRY(ov::opset8::Softmax),
            MAP_ENTRY(opset_latest::LogSoftmax),
            MAP_ENTRY(opset_latest::Tile),
            MAP_ENTRY(opset_latest::Split),
            MAP_ENTRY(opset_latest::Power),
            MAP_ENTRY(opset_latest::Multiply),
            MAP_ENTRY(opset_latest::Relu),
            MAP_ENTRY(opset_latest::Convolution),
            MAP_ENTRY(opset_latest::GroupConvolution),
            MAP_ENTRY(opset_latest::ConvolutionBackpropData),
            MAP_ENTRY(opset_latest::GroupConvolutionBackpropData),
            MAP_ENTRY(opset_latest::AvgPool),
            MAP_ENTRY(opset_latest::MaxPool),
            MAP_ENTRY(ov::opset8::AdaptiveAvgPool),
            MAP_ENTRY(ov::opset8::AdaptiveMaxPool),
            MAP_ENTRY(opset_latest::ShuffleChannels),
            MAP_ENTRY(ov::opset8::Gather),
            MAP_ENTRY(ov::opset8::GatherND),
            MAP_ENTRY(opset_latest::GatherTree),
            MAP_ENTRY(ov::opset8::NV12toRGB),
            MAP_ENTRY(ov::opset8::NV12toBGR),
            MAP_ENTRY(ov::opset8::I420toRGB),
            MAP_ENTRY(ov::opset8::I420toBGR),
            MAP_ENTRY(ov::opset8::RandomUniform),
            MAP_ENTRY(opset_latest::OneHot),
            MAP_ENTRY(opset_latest::BatchNormInference),
            MAP_ENTRY(opset_latest::GatherElements),
            MAP_ENTRY(opset_latest::ScatterNDUpdate),
            MAP_ENTRY(opset_latest::ScatterUpdate),
            MAP_ENTRY(opset_latest::ScatterElementsUpdate),
            MAP_ENTRY(opset_latest::Clamp),
            MAP_ENTRY(opset_latest::Elu),
            MAP_ENTRY(opset_latest::Reshape),
            MAP_ENTRY(opset_latest::Squeeze),
            MAP_ENTRY(opset_latest::Sigmoid),
            MAP_ENTRY(ov::opset9::GridSample),
            MAP_ENTRY(opset_latest::LRN),
            MAP_ENTRY(opset_latest::ReduceMax),
            MAP_ENTRY(opset_latest::ReduceMean),
            MAP_ENTRY(opset_latest::ReduceLogicalOr),
            MAP_ENTRY(opset_latest::ReduceLogicalAnd),
            MAP_ENTRY(opset_latest::ReduceProd),
            MAP_ENTRY(opset_latest::ReduceSum),
            MAP_ENTRY(opset_latest::ReduceMin),
            MAP_ENTRY(opset_latest::ReduceL1),
            MAP_ENTRY(opset_latest::ReduceL2),
            MAP_ENTRY(opset_latest::Unsqueeze),
            MAP_ENTRY(opset_latest::Minimum),
            MAP_ENTRY(opset_latest::Maximum),
            MAP_ENTRY(opset_latest::Add),
            MAP_ENTRY(opset_latest::Divide),
            MAP_ENTRY(opset_latest::SquaredDifference),
            MAP_ENTRY(opset_latest::FloorMod),
            MAP_ENTRY(opset_latest::Mod),
            MAP_ENTRY(opset_latest::Proposal),
            MAP_ENTRY(opset_latest::FakeQuantize),
            MAP_ENTRY(opset_latest::MatMul),
            MAP_ENTRY(opset_latest::Tan),
            MAP_ENTRY(opset_latest::Tanh),
            MAP_ENTRY(opset_latest::Sin),
            MAP_ENTRY(opset_latest::Cos),
            MAP_ENTRY(opset_latest::Sqrt),
            MAP_ENTRY(opset_latest::Sinh),
            MAP_ENTRY(opset_latest::Cosh),
            MAP_ENTRY(opset_latest::Asinh),
            MAP_ENTRY(opset_latest::Acosh),
            MAP_ENTRY(opset_latest::Atanh),
            MAP_ENTRY(opset_latest::Log),
            MAP_ENTRY(opset_latest::Selu),
            MAP_ENTRY(ov::opset2::Gelu),
            MAP_ENTRY(opset_latest::Exp),
            MAP_ENTRY(opset_latest::HSwish),
            MAP_ENTRY(opset_latest::Floor),
            MAP_ENTRY(opset_latest::Round),
            MAP_ENTRY(opset_latest::Mish),
            MAP_ENTRY(opset_latest::Erf),
            MAP_ENTRY(opset_latest::Broadcast),
            MAP_ENTRY(opset_latest::Bucketize),
            MAP_ENTRY(opset_latest::Transpose),
            MAP_ENTRY(opset_latest::Interpolate),
            MAP_ENTRY(opset_latest::TopK),
            MAP_ENTRY(ov::opset1::TopK),
            MAP_ENTRY(opset_latest::RegionYolo),
            MAP_ENTRY(opset_latest::ReorgYolo),
            MAP_ENTRY(ov::opset1::DetectionOutput),
            MAP_ENTRY(opset_latest::NormalizeL2),
            MAP_ENTRY(opset_latest::CumSum),
            MAP_ENTRY(ov::opset9::Eye),
            MAP_ENTRY(ov::opset4::MVN),
            MAP_ENTRY(ov::opset6::MVN),
            MAP_ENTRY(opset_latest::Concat),
            MAP_ENTRY(opset_latest::ROIPooling),
            MAP_ENTRY(opset_latest::PSROIPooling),
            MAP_ENTRY(ov::op::v9::ROIAlign),
            MAP_ENTRY(opset_latest::StridedSlice),
            MAP_ENTRY(opset_latest::PRelu),
            MAP_ENTRY(opset_latest::Swish),
            MAP_ENTRY(opset_latest::GRN),
            MAP_ENTRY(opset_latest::Negative),
            MAP_ENTRY(opset_latest::Sign),
            MAP_ENTRY(opset_latest::CTCGreedyDecoder),
            MAP_ENTRY(opset_latest::CTCGreedyDecoderSeqLen),
            MAP_ENTRY(ov::opset1::Pad),
            MAP_ENTRY(opset_latest::LSTMCell),
            MAP_ENTRY(opset_latest::Subtract),
            MAP_ENTRY(opset_latest::LogicalAnd),
            MAP_ENTRY(opset_latest::LSTMSequence),
            MAP_ENTRY(opset_latest::Ceiling),
            MAP_ENTRY(opset_latest::SoftPlus),
            MAP_ENTRY(opset_latest::Equal),
            MAP_ENTRY(opset_latest::Select),
            MAP_ENTRY(ov::opset9::NonMaxSuppression),
            MAP_ENTRY(ov::op::internal::NonMaxSuppressionIEInternal),
            MAP_ENTRY(opset_latest::DepthToSpace),
            MAP_ENTRY(opset_latest::ReverseSequence),
            MAP_ENTRY(opset_latest::Less),
            MAP_ENTRY(opset_latest::LessEqual),
            MAP_ENTRY(opset_latest::NotEqual),
            MAP_ENTRY(opset_latest::Greater),
            MAP_ENTRY(opset_latest::GreaterEqual),
            MAP_ENTRY(opset_latest::LogicalNot),
            MAP_ENTRY(opset_latest::LogicalOr),
            MAP_ENTRY(opset_latest::LogicalXor),
            MAP_ENTRY(opset_latest::SpaceToDepth),
            MAP_ENTRY(opset_latest::SpaceToBatch),
            MAP_ENTRY(opset_latest::ExtractImagePatches),
            MAP_ENTRY(opset_latest::Abs),
            MAP_ENTRY(opset_latest::Atan),
            MAP_ENTRY(opset_latest::Asin),
            MAP_ENTRY(opset_latest::Acos),
            MAP_ENTRY(opset_latest::Roll),
            MAP_ENTRY(opset_latest::HSigmoid),
            MAP_ENTRY(opset_latest::HardSigmoid),
            MAP_ENTRY(opset_latest::EmbeddingBagOffsetsSum),
            MAP_ENTRY(opset_latest::EmbeddingSegmentsSum),
            MAP_ENTRY(opset_latest::EmbeddingBagPackedSum),
            MAP_ENTRY(ov::opset3::Assign),
            MAP_ENTRY(ov::opset3::ReadValue),
            MAP_ENTRY(ov::opset6::Assign),
            MAP_ENTRY(ov::opset6::ReadValue),
            MAP_ENTRY(opset_latest::GRUCell),
            MAP_ENTRY(opset_latest::GRUSequence),
            MAP_ENTRY(opset_latest::DeformablePSROIPooling),
            MAP_ENTRY(opset_latest::DFT),
            MAP_ENTRY(opset_latest::IDFT),
            MAP_ENTRY(ov::opset9::RDFT),
            MAP_ENTRY(ov::opset9::IRDFT),
            MAP_ENTRY(ov::opset8::If),
    };

#undef MAP_ENTRY

    const auto dispatchIt = dispatchMap.find(op->get_type_info());
    return (dispatchIt != dispatchMap.end()) ? dispatchIt->second : nullptr;
}

// TODO Extend implementation to check architecture, limitation and can we really compile it
bool NGraphImporter::isOpSupported(const std::shared_ptr<ov::Node>& op) {
    const bool hasParser = (NGraphImporter::getParser(op) != nullptr);
    return hasParser;
}

mlir::Type importPrecision(mlir::MLIRContext* ctx, const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::f64:
        return mlir::Float64Type::get(ctx);
    case ov::element::Type_t::f32:
        return mlir::Float32Type::get(ctx);
    case ov::element::Type_t::f16:
        return mlir::Float16Type::get(ctx);
    case ov::element::Type_t::bf16:
        return mlir::BFloat16Type::get(ctx);
    case ov::element::Type_t::i64:
        return getSInt64Type(ctx);
    case ov::element::Type_t::u64:
        return getUInt64Type(ctx);
    case ov::element::Type_t::i32:
        return getSInt32Type(ctx);
    case ov::element::Type_t::u32:
        return getUInt32Type(ctx);
    case ov::element::Type_t::i16:
        return getSInt16Type(ctx);
    case ov::element::Type_t::u16:
        return getUInt16Type(ctx);
    case ov::element::Type_t::i8:
        return getSInt8Type(ctx);
    case ov::element::Type_t::u8:
        return getUInt8Type(ctx);
    case ov::element::Type_t::boolean:
        return getBool8Type(ctx);
    default:
        VPUX_THROW("Unsupported precision : '{0}'", precision);
    }
}

//
// buildMainFunc
//

mlir::func::FuncOp NGraphImporter::buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName,
                                                 mlir::TimingScope& rootTiming, bool stubLayers) {
    auto scopeTiming = rootTiming.nest("Import nGraph function");

    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(_netGraph->get_parameters().size());
    for (const auto& param : _netGraph->get_parameters()) {
        inputTypes.push_back(importTensor(param->get_partial_shape(), param->get_element_type()));
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0).size() == 0
                                                   ? ov::PartialShape{1}
                                                   : result->get_input_partial_shape(0),
                                           result->get_input_element_type(0)));
    }

    const auto funcType = mlir::FunctionType::get(_ctx, ArrayRef(inputTypes), ArrayRef(outputTypes));

    auto func = moduleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_ctx), funcName, funcType);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    for (const auto& p : _netGraph->get_parameters() | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = checked_cast<uint32_t>(p.index());

        _log.trace("Convert network Parameter {0}", paramNode->get_friendly_name());

        const auto funcInputVal = func.getArgument(paramIndex);
        _importedVals.emplace(paramNode->output(0), funcInputVal);
    }

    for (const auto& origNode : _netGraph->get_ordered_ops()) {
        _log.trace("Convert {0} layer {1}", origNode->get_type_name(), origNode->get_friendly_name());
        const auto parser = NGraphImporter::getParser(origNode);

        if (parser != nullptr) {
            (this->*parser)(builder, origNode);
        } else {
            VPUX_THROW_UNLESS(stubLayers,
                              "[NOT IMPLEMENTED] Unsupported operation {0} with type {1}. "
                              "Try to update the driver to the latest version. If the error persists, "
                              "please submit a bug report in https://github.com/openvinotoolkit/openvino/issues",
                              origNode->get_friendly_name(), origNode->get_type_name());
            parseNodeAsStub(builder, origNode);
        }
    }

    SmallVector<mlir::Value> funcOutputs;
    funcOutputs.reserve(_netGraph->get_results().size());

    for (const auto& p : _netGraph->get_results() | indexed) {
        const auto& resultNode = p.value();

        _log.trace("Convert network Result {0}", resultNode->get_friendly_name());

        const auto resultInputs = getInputs(resultNode);
        VPUX_THROW_UNLESS(resultInputs.size() == 1, "nGraph Result {0} has unsupported number of inputs {1}",
                          resultNode->get_friendly_name(), resultInputs.size());

        funcOutputs.push_back(resultInputs[0]);
    }

    SmallVector<mlir::NamedAttribute> fields;
    fields.emplace_back(mlir::StringAttr::get(_ctx, "name"), mlir::StringAttr::get(_ctx, "output"));
    fields.emplace_back(mlir::StringAttr::get(_ctx, "type"), mlir::StringAttr::get(_ctx, "Output"));
    auto metadata = mlir::DictionaryAttr::get(_ctx, fields);
    auto retLoc = mlir::FusedLoc::get(_ctx, {mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "output"))}, metadata);

    builder.create<mlir::func::ReturnOp>(retLoc, ArrayRef(funcOutputs));

    return func;
}

void NGraphImporter::buildBlockFromRegion(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block) {
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(_netGraph->get_parameters().size());
    for (const auto& param : _netGraph->get_parameters()) {
        inputTypes.push_back(importTensor(param->get_partial_shape(), param->get_element_type()));
    }
    SmallVector<mlir::Location> inputTypeLocations(inputTypes.size(), loc);

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }
    SmallVector<mlir::Location> outputTypeLocations(outputTypes.size(), loc);

    block->addArguments(inputTypes, inputTypeLocations);

    for (const auto& p : _netGraph->get_parameters() | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = checked_cast<uint32_t>(p.index());

        _log.trace("Convert network Parameter {0}", paramNode->get_friendly_name());

        const auto funcInputVal = block->getArgument(paramIndex);
        _importedVals.emplace(paramNode->output(0), funcInputVal);
    }

    for (const auto& origNode : _netGraph->get_ordered_ops()) {
        _log.trace("Convert {0} layer {1}", origNode->get_type_name(), origNode->get_friendly_name());
        const auto parser = NGraphImporter::getParser(origNode);

        if (parser != nullptr) {
            (this->*parser)(builder, origNode);
        } else {
            parseNodeAsStub(builder, origNode);
        }
    }
    SmallVector<mlir::Value> blockOutputs;
    blockOutputs.reserve(_netGraph->get_results().size());
    for (const auto& p : _netGraph->get_results() | indexed) {
        const auto& resultNode = p.value();

        _log.trace("Convert network Result {0}", resultNode->get_friendly_name());

        const auto resultInputs = getInputs(resultNode);
        VPUX_THROW_UNLESS(resultInputs.size() == 1, "nGraph Result {0} has unsupported number of inputs {1}",
                          resultNode->get_friendly_name(), resultInputs.size());

        blockOutputs.push_back(resultInputs[0]);
    }
    builder.create<IE::YieldOp>(loc, blockOutputs);
    return;
}

SmallVector<mlir::Type> NGraphImporter::getRegionResults() {
    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }
    return outputTypes;
}

//
// Parsers
//

void NGraphImporter::parseNodeAsStub(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    _log.debug("{0} is not supported. Replacing with a Stub layer", origNode->get_friendly_name());

    const auto inputs = getInputs(origNode);
    const auto outputCount = static_cast<int32_t>(origNode->get_output_size());

    SmallVector<mlir::Type> outputTypes;
    for (auto i = 0; i < outputCount; ++i) {
        auto type = importTensor(origNode->get_output_shape(i), origNode->get_output_element_type(i));
        outputTypes.push_back(type);
    }

    auto op = builder.create<IE::StubOp>(createLocation(origNode), outputTypes, inputs);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Constant>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Constant>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.empty(), "nGraph Constant node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto tensorType = importTensor(origNode->get_output_partial_shape(0), origNode->get_output_element_type(0));

    const auto numElems = tensorType.getNumElements();
    const Byte elemTypeSize = getElemTypeSize(tensorType).to<Byte>();

    auto value = [&]() -> mlir::ElementsAttr {
        const auto rawBuffer = ArrayRef(origNode->get_data_ptr<char>(), numElems * elemTypeSize.count());
        if (!_sharedConstants) {
            return mlir::DenseElementsAttr::getFromRawBuffer(tensorType, rawBuffer);
        }

        constexpr size_t defaultAlignment =
                alignof(std::max_align_t);  // seemingly used nowhere except no-op deleter - use C++ default
        constexpr auto noopDeleter = [](void*, size_t, size_t) {};
        constexpr bool isMutable = false;
        mlir::AsmResourceBlob blob(rawBuffer, defaultAlignment, noopDeleter, isMutable);

        auto& builtinDialectManager = mlir::DenseResourceElementsHandle::getManagerInterface(builder.getContext());
        // assumption (as per MLIR documented behavior): inserting a new blob with the same key would internally cause
        // the key to change, so that there are no collisions - thus, the blob is never overwritten here
        return mlir::DenseResourceElementsAttr::get(
                tensorType, builtinDialectManager.insert("ngraphSharedConstant", std::move(blob)));
    }();

    auto op = builder.create<Const::DeclareOp>(createLocation(origNode), tensorType, Const::ContentAttr::get(value));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convert>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Convert>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Convert node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto dstType = importPrecision(_ctx, origNode->get_destination_type());
    const auto dstTypeAttr = mlir::TypeAttr::get(dstType);

    auto op = builder.create<IE::ConvertOp>(createLocation(origNode), inputs[0], dstTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Softmax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::Softmax>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Softmax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getIntAttr(_ctx, axis);

    auto op = builder.create<IE::SoftMaxOp>(createLocation(origNode), inputs[0], axisAttr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogSoftmax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, opset_latest::LogSoftmax>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph LogSoftmax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getIntAttr(_ctx, axis);

    auto op = builder.create<IE::LogSoftmaxOp>(createLocation(origNode), inputs[0], axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tile>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Tile>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Tile node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::TileOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Relu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Relu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ReLUOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Split>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Split>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Split node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto num_splits = origNode->get_num_splits();
    const auto numSplitsAttr = getIntAttr(_ctx, num_splits);

    auto op = builder.create<IE::SplitOp>(createLocation(origNode), inputs[0], inputs[1], numSplitsAttr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Power>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Power>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Power node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::PowerOp>(createLocation(origNode), inputs[0], inputs[1],
                                          importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Multiply>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Multiply>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Multiply node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MultiplyOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type), nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MatMul>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::MatMul>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::MatMulOp>(createLocation(origNode), inputs[0], inputs[1], origNode->get_transpose_a(),
                                           origNode->get_transpose_b());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convolution>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Convolution>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::ConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, attrStride,
                                                attrPadsBegin, attrPadsEnd, attrDilation, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::GroupConvolution>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::GroupConvolution>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::GroupConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr,
                                                     attrStride, attrPadsBegin, attrPadsEnd, attrDilation,
                                                     /*groups=*/nullptr,
                                                     /*post_op=*/nullptr,
                                                     /*clamp=*/nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ConvolutionBackpropData>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ConvolutionBackpropData>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 2) || (inputs.size() == 3),
                      "nGraph node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrOutputPadding = getIntArrayAttr(_ctx, origNode->get_output_padding());

    auto optionalOutputShapeInput = inputs.size() == 2 ? nullptr : inputs[2];
    auto op = builder.create<IE::ConvolutionBackpropDataOp>(createLocation(origNode), inputs[0], inputs[1],
                                                            optionalOutputShapeInput, attrStride, attrPadsBegin,
                                                            attrPadsEnd, attrDilation, attrOutputPadding);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::GroupConvolutionBackpropData>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::GroupConvolutionBackpropData>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 2) || (inputs.size() == 3),
                      "nGraph node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrOutputPadding = getIntArrayAttr(_ctx, origNode->get_output_padding());

    auto optionalOutputShapeInput = inputs.size() == 2 ? nullptr : inputs[2];
    auto op = builder.create<IE::GroupConvolutionBackpropDataOp>(createLocation(origNode), inputs[0], inputs[1],
                                                                 optionalOutputShapeInput, attrStride, attrPadsBegin,
                                                                 attrPadsEnd, attrDilation, attrOutputPadding);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::AvgPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::AvgPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());
    const auto attrExcludePads = origNode->get_exclude_pad() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::AvgPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType, attrExcludePads, nullptr,
                                            nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::MaxPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());

    auto op = builder.create<IE::MaxPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveAvgPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::AdaptiveAvgPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AdaptiveAvgPoolOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveMaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::AdaptiveMaxPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto dstTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));

    auto op = builder.create<IE::AdaptiveMaxPoolOp>(createLocation(origNode), inputs[0], inputs[1], dstTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Add>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Add>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op =
            builder.create<IE::AddOp>(createLocation(origNode), inputs[0], inputs[1], importBroadcastType(autob.m_type),
                                      /*post_op=*/nullptr, /*clamp=*/nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Divide>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Divide>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::DivideOp>(createLocation(origNode), inputs[0], inputs[1],
                                           importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::SquaredDifference>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::SquaredDifference>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SquaredDifferenceOp>(createLocation(origNode), inputs[0], inputs[1],
                                                      importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FloorMod>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::FloorMod>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::FloorModOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Mod>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Mod>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::ModOp>(createLocation(origNode), inputs[0], inputs[1],
                                        importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ShuffleChannels>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ShuffleChannels>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ShuffleChannels node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ShuffleChannelsOp>(createLocation(origNode), inputs[0], origNode->get_axis(),
                                                    origNode->get_group());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Gather>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::Gather>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Gather node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto batchDims = origNode->get_batch_dims();
    auto idxRank = origNode->get_input_partial_shape(1).rank().get_length();
    batchDims = (batchDims < 0) ? (batchDims + idxRank) : batchDims;
    auto normBatchDims = getIntAttr(_ctx, batchDims);
    VPUX_THROW_UNLESS(batchDims >= 0, "Invalid batch_dims value '{0}'", batchDims);

    auto op = builder.create<IE::GatherOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                           normBatchDims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::GatherND>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::GatherND>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph GatherND node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto batchDims = origNode->get_batch_dims();

    auto op = builder.create<IE::GatherNDOp>(createLocation(origNode), inputs[0], inputs[1], batchDims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GatherTree>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, opset_latest::GatherTree>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph GatherTree node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::GatherTreeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toRGB>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::NV12toRGB>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2 && !inputs.empty(),
                      "nGraph NV12toRGB node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, nullptr,
                                             IE::ColorFmt::NV12, IE::ColorFmt::RGB);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toBGR>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::NV12toBGR>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2 && !inputs.empty(),
                      "nGraph NV12toBGR node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, nullptr,
                                             IE::ColorFmt::NV12, IE::ColorFmt::BGR);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toRGB>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::I420toRGB>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 1) || (inputs.size() == 3),
                      "nGraph I420toRGB node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto thirdInput = inputs.size() == 1 ? nullptr : inputs[2];
    IE::YuvToRgbOp op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, thirdInput,
                                                       IE::ColorFmt::I420, IE::ColorFmt::RGB);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toBGR>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::I420toBGR>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 1) || (inputs.size() == 3),
                      "nGraph I420toBGR node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto thirdInput = inputs.size() == 1 ? nullptr : inputs[2];
    IE::YuvToRgbOp op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, thirdInput,
                                                       IE::ColorFmt::I420, IE::ColorFmt::BGR);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::RandomUniform>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::RandomUniform>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph RandomUniform node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto shapeConstant = dynamic_cast<opset_latest::Constant*>(origNode->input_value(0).get_node());
    VPUX_THROW_UNLESS(shapeConstant != nullptr,
                      "nGraph RandomUniform node '{0}' must have Constant shape input in order to infer output shape.",
                      origNode->get_friendly_name());
    const auto shapeValues = shapeConstant->cast_vector<int64_t>();
    const auto shapeAttr = getIntArrayAttr(_ctx, shapeValues);

    const auto outputTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_input_element_type(1)));
    const auto globalSeedAttr = getIntAttr(_ctx, origNode->get_global_seed());
    const auto opSeedAttr = getIntAttr(_ctx, origNode->get_op_seed());

    auto op = builder.create<IE::RandomUniformOp>(createLocation(origNode), inputs[1], inputs[2], shapeAttr,
                                                  outputTypeAttr, globalSeedAttr, opSeedAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::OneHot>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::OneHot>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph OneHot node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto outElemTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_element_type()));

    IE::OneHotOp op = builder.create<IE::OneHotOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                   nullptr, nullptr, nullptr, axisAttr, outElemTypeAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::BatchNormInference>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::BatchNormInference>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 5, "nGraph BatchNorm node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto epsAttr = getFPAttr(_ctx, origNode->get_eps_value());
    auto op =
            builder.create<IE::BatchNormInferenceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                     inputs[3], inputs[4], nullptr, nullptr, nullptr, nullptr, epsAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::GatherElements>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::GatherElements>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph GatherElements node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto origAxis = origNode->get_axis();
    auto rank = origNode->get_input_partial_shape(0).rank().get_length();
    origAxis = (origAxis < 0) ? (origAxis + rank) : origAxis;
    auto axis = getIntAttr(_ctx, origAxis);

    auto op = builder.create<IE::GatherElementsOp>(createLocation(origNode), inputs[0], inputs[1], axis);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ScatterNDUpdate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ScatterNDUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph ScatterNDUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ScatterNDUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ScatterUpdate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ScatterUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph ScatterUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ScatterUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                  nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ScatterElementsUpdate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ScatterElementsUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4,
                      "nGraph ScatterElementsUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ScatterElementsUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                          inputs[3], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Reshape>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Reshape>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Reshape node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ReshapeOp>(createLocation(origNode), inputs[0], inputs[1],
                                            origNode->get_special_zero(), nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Minimum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Minimum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Minimum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MinimumOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Maximum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Maximum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Maximum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MaximumOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Clamp>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Clamp>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Clamp node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto min = origNode->get_min();
    const auto max = origNode->get_max();
    const auto minAttr = getFPAttr(_ctx, min);
    const auto maxAttr = getFPAttr(_ctx, max);

    auto op = builder.create<IE::ClampOp>(createLocation(origNode), inputs[0], minAttr, maxAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Proposal>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Proposal>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Proposal node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& proposalParam = origNode->get_attrs();
    const auto proposalParamAttr = importProposalAttrs(proposalParam);

    auto op = builder.create<IE::ProposalOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                             proposalParamAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Unsqueeze>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Unsqueeze>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::UnsqueezeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LRN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::LRN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto beta = origNode->get_beta();
    const auto bias = origNode->get_bias();
    const auto size = origNode->get_nsize();

    const auto alphaAttr = getFPAttr(_ctx, alpha);
    const auto betaAttr = getFPAttr(_ctx, beta);
    const auto biasAttr = getFPAttr(_ctx, bias);
    const auto sizeAttr = getIntAttr(_ctx, size);

    auto op = builder.create<IE::LRNOp>(createLocation(origNode), inputs[0], inputs[1], alphaAttr, betaAttr, biasAttr,
                                        sizeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Broadcast>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Broadcast>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                      "nGraph Broadcast node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto mode = importBroadcastMode(origNode->get_broadcast_spec().m_type);

    IE::BroadcastOp op;
    if (inputs.size() == 2) {
        op = builder.create<IE::BroadcastOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, mode);
    } else {
        op = builder.create<IE::BroadcastOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], mode);
    }

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Bucketize>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Bucketize>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Bucketize node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_output_type()));
    const auto withRightBoundAttr = origNode->get_with_right_bound() ? mlir::UnitAttr::get(_ctx) : nullptr;

    VPUX_THROW_UNLESS(outputTypeAttr == mlir::TypeAttr::get(importPrecision(_ctx, ov::element::i32)),
                      "nGraph Bucketize output_type attribute '{0}' should have only SI32 type", outputTypeAttr);

    auto op = builder.create<IE::BucketizeOp>(createLocation(origNode), inputs[0], inputs[1], outputTypeAttr,
                                              withRightBoundAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceMax>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMaxOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMean>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceMean>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMean node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMeanOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ReduceLogicalOr>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceLogicalOr>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceLogicalOr node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceLogicalOrOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ReduceLogicalAnd>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceLogicalAnd>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceLogicalAnd node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op =
            builder.create<IE::ReduceLogicalAndOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceProd>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceProd>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceProd node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceProdOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceSum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceSumOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMin>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceMin>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMin node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMinOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceL1>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::ReduceL1>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceL1 node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceL1Op>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceL2>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::ReduceL2>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceL2 node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceL2Op>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sigmoid>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SigmoidOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::GridSample>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::GridSample>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    const auto alignCornersAttr = origNode->get_attributes().align_corners ? mlir::UnitAttr::get(_ctx) : nullptr;
    const auto modeAttr = importGridSampleMode(origNode->get_attributes().mode);
    const auto paddingModeAttr = importGridSamplePaddingMode(origNode->get_attributes().padding_mode);

    auto op = builder.create<IE::GridSampleOp>(createLocation(origNode), inputs[0], inputs[1], alignCornersAttr,
                                               modeAttr, paddingModeAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Squeeze>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Squeeze>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2 && inputs.size() > 0,
                      "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());
    if (inputs.size() == 2) {
        auto op = builder.create<IE::SqueezeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
        addOutputs(origNode, op);
    } else {
        auto op = builder.create<IE::SqueezeOp>(createLocation(origNode), inputs[0], nullptr, nullptr);
        addOutputs(origNode, op);
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Transpose>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Transpose>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Transpose node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TransposeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tan>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Tan>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Tan node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TanOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tanh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Tanh>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Tanh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TanhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sin>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sin>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sin node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SinOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Cos>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Cos>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Cos node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CosOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sqrt>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sqrt>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sqrt node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SqrtOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sinh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sinh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sinh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SinhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Cosh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Cosh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Cosh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CoshOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asinh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Asinh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Asinh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AsinhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acosh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Acosh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Acosh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AcoshOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atanh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Atanh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Atanh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AtanhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Roll>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v7::Roll>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Roll node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RollOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Log>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Log>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Log node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::LogOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Selu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Selu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Selu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SeluOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr, nullptr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::Gelu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Gelu>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Gelu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::GeluOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Elu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Elu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Elu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto alphaAttr = getFPAttr(_ctx, alpha);

    auto op = builder.create<IE::EluOp>(createLocation(origNode), inputs[0], alphaAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HSwish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::HSwish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph HSwish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HSwishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Floor>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Floor>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Floor node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::FloorOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Round>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::Round>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Round node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RoundOp>(createLocation(origNode), inputs[0], importRoundMode(origNode->get_mode()));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Mish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Mish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Mish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::MishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Erf>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Erf>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Erf node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ErfOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FakeQuantize>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::FakeQuantize>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 5, "nGraph FakeQuantize node '{0}' has unsupported number of inputs '{1}'.",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_auto_broadcast();

    const auto levelsAttr = getIntAttr(_ctx, origNode->get_levels());

    auto op = builder.create<IE::FakeQuantizeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 inputs[4], levelsAttr, importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Exp>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Exp>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Exp node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ExpOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::StridedSlice>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::StridedSlice>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph StridedSlice node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto attrBeginMask = getIntArrayAttr(_ctx, origNode->get_begin_mask());
    auto attrEndMask = getIntArrayAttr(_ctx, origNode->get_end_mask());
    auto attrNewAxisMask = getIntArrayAttr(_ctx, origNode->get_new_axis_mask());
    auto attrShrinkAxisMask = getIntArrayAttr(_ctx, origNode->get_shrink_axis_mask());
    auto attrEllipsisAxisMask = getIntArrayAttr(_ctx, origNode->get_ellipsis_mask());

    auto op = builder.create<IE::StridedSliceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 nullptr, nullptr, nullptr, attrBeginMask, attrEndMask, attrNewAxisMask,
                                                 attrShrinkAxisMask, attrEllipsisAxisMask);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ROIPooling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ROIPooling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ROIPooling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputSize = getIntArrayAttr(_ctx, origNode->get_output_size());
    const auto spatialScaleAttr = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto method = importROIPoolingMethod(origNode->get_method());

    auto op = builder.create<IE::ROIPoolingOp>(createLocation(origNode), inputs[0], inputs[1], outputSize,
                                               spatialScaleAttr, method);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v9::ROIAlign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::ROIAlign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph ROIAlign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto pooled_h = getIntAttr(_ctx, origNode->get_pooled_h());
    const auto pooled_w = getIntAttr(_ctx, origNode->get_pooled_w());
    const auto sampling_ratio = getIntAttr(_ctx, origNode->get_sampling_ratio());
    const auto spatialScaleAttr = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto poolingMode = importROIAlignMethod(origNode->get_mode());
    const auto alignedMode = importROIAlignAlignedMethod(origNode->get_aligned_mode());

    auto op = builder.create<IE::ROIAlignOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], pooled_h,
                                             pooled_w, sampling_ratio, spatialScaleAttr, poolingMode, alignedMode);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Concat>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Concat>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() >= 1, "nGraph Concat node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getIntAttr(_ctx, axis);

    auto op = builder.create<IE::ConcatOp>(createLocation(origNode), inputs, axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Interpolate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Interpolate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 4) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto interpolateAttr = importInterpolateAttrs(origNode->get_attrs());

    IE::InterpolateOp op;
    if (inputs.size() == 3) {
        op = builder.create<IE::InterpolateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, interpolateAttr);
    } else {
        op = builder.create<IE::InterpolateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, interpolateAttr);
    }

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v3::TopK>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::TopK>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph TopK node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto modeAttr = importTopKMode(origNode->get_mode());
    const auto sortTypeAttr = importTopKSortType(origNode->get_sort_type());
    const auto indexElementTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));

    auto op = builder.create<IE::TopKOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, axisAttr, modeAttr,
                                         sortTypeAttr, indexElementTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v1::TopK>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::TopK>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph TopK node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto modeAttr = importTopKMode(origNode->get_mode());
    const auto sortTypeAttr = importTopKSortType(origNode->get_sort_type());
    const auto indexElementTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));

    auto op = builder.create<IE::TopKOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, axisAttr, modeAttr,
                                         sortTypeAttr, indexElementTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::RegionYolo>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::RegionYolo>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph RegionYolo node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto coordAttr = getIntAttr(_ctx, origNode->get_num_coords());
    const auto classesAttr = getIntAttr(_ctx, origNode->get_num_classes());
    const auto regionsAttr = getIntAttr(_ctx, origNode->get_num_regions());
    const auto doSoftmaxAttr = mlir::BoolAttr::get(_ctx, origNode->get_do_softmax());
    const auto maskAttr = getIntArrayAttr(_ctx, origNode->get_mask());
    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto axisEndAttr = getIntAttr(_ctx, origNode->get_end_axis());
    const auto anchorsAttr = getFPArrayAttr(_ctx, origNode->get_anchors());

    auto op = builder.create<IE::RegionYoloOp>(createLocation(origNode), inputs[0], coordAttr, classesAttr, regionsAttr,
                                               doSoftmaxAttr, maskAttr, axisAttr, axisEndAttr, anchorsAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReorgYolo>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ReorgYolo>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ReorgYolo node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto strides = origNode->get_strides();

    VPUX_THROW_UNLESS(strides.size() == 2, "nGraph ReorgYolo node '{0}' has unsupported number of strides '{1}'",
                      origNode->get_friendly_name(), strides.size());
    VPUX_THROW_UNLESS(strides.front() == strides.back(),
                      "nGraph ReorgYolo node '{0}' has different strides '{1}' != '{2}'", origNode->get_friendly_name(),
                      strides.front(), strides.back());

    const auto strideAttr = getIntAttr(_ctx, strides.front());

    auto op = builder.create<IE::ReorgYoloOp>(createLocation(origNode), inputs[0], strideAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DetectionOutput>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::DetectionOutput>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3 || inputs.size() == 5,
                      "nGraph DetectionOutput node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto detectionOutputAttr = importDetectionOutputAttrs(origNode->get_attrs());

    IE::DetectionOutputOp op;
    if (inputs.size() == 3) {
        op = builder.create<IE::DetectionOutputOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                                   nullptr, detectionOutputAttr);
    } else {
        op = builder.create<IE::DetectionOutputOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                   inputs[4], detectionOutputAttr);
    }
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::NormalizeL2>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::NormalizeL2>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph NormalizeL2 node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto epsAttr = getFPAttr(_ctx, origNode->get_eps());
    const auto epsModeAttr = importEpsMode(origNode->get_eps_mode());

    auto op = builder.create<IE::NormalizeL2Op>(createLocation(origNode), inputs[0], inputs[1], nullptr, epsAttr,
                                                epsModeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::CumSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, opset_latest::CumSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 1 || inputs.size() == 2,
                      "nGraph CumSum node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto exclusiveAttr = origNode->is_exclusive() ? mlir::UnitAttr::get(_ctx) : nullptr;
    const auto reverseAttr = origNode->is_reverse() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::CumSumOp>(createLocation(origNode), inputs[0],
                                           (inputs.size() == 2) ? inputs[1] : nullptr, nullptr, exclusiveAttr,
                                           reverseAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::Eye>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::Eye>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 3 || inputs.size() == 4,
                      "nGraph Eye node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto outputTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_out_type()));

    auto op = builder.create<IE::EyeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                        (inputs.size() == 4) ? inputs[3] : nullptr, nullptr, nullptr, nullptr,
                                        outputTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::MVN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::MVN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph MVN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto normalize_varianceAttr = mlir::BoolAttr::get(_ctx, origNode->get_normalize_variance());
    const auto across_channelsAttr = mlir::BoolAttr::get(_ctx, origNode->get_across_channels());
    const auto epsAttr = getFPAttr(_ctx, origNode->get_eps());

    auto op = builder.create<IE::MVNOp>(createLocation(origNode), inputs[0], across_channelsAttr,
                                        normalize_varianceAttr, epsAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::MVN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::MVN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph MVN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto normalize_varianceAttr = mlir::BoolAttr::get(_ctx, origNode->get_normalize_variance());

    const auto epsAttr = getFPAttr(_ctx, origNode->get_eps());
    const auto epsModeAttr = importMvnEpsMode(origNode->get_eps_mode());

    auto op = builder.create<IE::MVN6Op>(createLocation(origNode), inputs[0], inputs[1], nullptr,
                                         normalize_varianceAttr, epsAttr, epsModeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::PRelu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::PRelu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph PRelu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::PReluOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Swish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Swish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    vpux::IE::SwishOp op;

    if (inputs.size() == 1) {
        op = builder.create<IE::SwishOp>(createLocation(origNode), inputs[0], nullptr, nullptr);
    } else if (inputs.size() == 2) {
        op = builder.create<IE::SwishOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    } else {
        VPUX_THROW("nGraph Swish node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                   inputs.size());
    }

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::GRN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph GRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto biasAttr = getFPAttr(_ctx, origNode->get_bias());

    auto op = builder.create<IE::GRNOp>(createLocation(origNode), inputs[0], biasAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Negative>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Negative>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Negative node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::NegativeOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SignOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::CTCGreedyDecoder>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::CTCGreedyDecoder>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph CTCGreedyDecoder node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CTCGreedyDecoderOp>(createLocation(origNode), inputs[0], inputs[1],
                                                     origNode->get_ctc_merge_repeated());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::CTCGreedyDecoderSeqLen>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::CTCGreedyDecoderSeqLen>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    if (inputs.size() == 3) {
        auto op = builder.create<IE::CTCGreedyDecoderSeqLenOp>(createLocation(origNode), inputs[0], inputs[1],
                                                               inputs[2], origNode->get_merge_repeated());
        addOutputs(origNode, op);
    } else if (inputs.size() == 2) {
        auto op = builder.create<IE::CTCGreedyDecoderSeqLenOp>(createLocation(origNode), inputs[0], inputs[1],
                                                               /*blankIndex*/ nullptr, origNode->get_merge_repeated());
        addOutputs(origNode, op);
    } else {
        VPUX_THROW("nGraph CTCGreedyDecoderSeqLen node '{0}' has unsupported number of inputs '{1}'",
                   origNode->get_friendly_name(), inputs.size());
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v1::Pad>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Pad>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    if (inputs.size() == 4) {
        auto op = builder.create<IE::PadOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                            nullptr, nullptr, nullptr, importPadMode(origNode->get_pad_mode()));
        addOutputs(origNode, op);
    } else if (inputs.size() == 3) {
        auto op = builder.create<IE::PadOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr, nullptr,
                                            nullptr, nullptr, importPadMode(origNode->get_pad_mode()));
        addOutputs(origNode, op);
    } else {
        VPUX_THROW("nGraph Pad node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                   inputs.size());
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LSTMCell>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::LSTMCell>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 6, "nGraph LSTMCell node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph LSTMCell node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh", "tanh"}),
                      "nGraph LSTMCell node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto hiddenSizeAttr = getIntAttr(_ctx, origNode->get_hidden_size());

    auto op = builder.create<IE::LSTMCellOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                             inputs[4], inputs[5], hiddenSizeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LSTMSequence>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, opset_latest::LSTMSequence>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 7, "nGraph LSTMSequence node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph LSTMSequence node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh", "tanh"}),
                      "nGraph LSTMSequence node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    VPUX_THROW_UNLESS(origNode->get_direction() != opset_latest::LSTMSequence::direction::BIDIRECTIONAL,
                      "nGraph LSTMSequence node '{0}' has unsupported direction 'BIDIRECTIONAL'",
                      origNode->get_friendly_name());
    const auto directionAttr = importRNNSequenceDirection(origNode->get_direction());

    const auto seqLenConstant = dynamic_cast<opset_latest::Constant*>(origNode->input_value(3).get_node());
    VPUX_THROW_UNLESS(
            seqLenConstant != nullptr,
            "nGraph LSTMSequence node '{0}' has unsupported sequenceLengths input. It must be a Constant node",
            origNode->get_friendly_name());
    const auto seqLenValues = seqLenConstant->cast_vector<uint32_t>();
    VPUX_THROW_UNLESS(seqLenValues.size() > 0,
                      "nGraph LSTMSequence node '{0}' has unsupported sequenceLengths input. It must contain more than "
                      "0 elements",
                      origNode->get_friendly_name());
    const auto isAllLensSame =
            std::all_of(seqLenValues.cbegin(), seqLenValues.cend(), [&seqLenValues](const auto item) {
                return seqLenValues[0] == item;
            });
    VPUX_THROW_UNLESS(
            isAllLensSame,
            "nGraph LSTMSequence node '{0}' has unsupported sequenceLengths input. It must contain all the same values",
            origNode->get_friendly_name());
    const auto seqLenAttr = getIntAttr(_ctx, checked_cast<uint32_t>(seqLenValues[0]));

    auto op = builder.create<IE::LSTMSequenceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[4],
                                                 inputs[5], inputs[6], seqLenAttr, directionAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Subtract>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Subtract>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SubtractOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type), nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalAnd>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, opset_latest::LogicalAnd>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::AndOp>(createLocation(origNode), inputs[0], inputs[1],
                                        importBroadcastType(autob.m_type), nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Ceiling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Ceiling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Ceiling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CeilingOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Equal>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Equal>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::EqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                          importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Select>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Select>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Select node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SelectOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                           importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset9::NonMaxSuppression>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset9::NonMaxSuppression>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 5) || (inputs.size() == 6),
                      "nGraph NonMaxSuppression node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto softNmsSigma = inputs.size() == 6 ? inputs[5] : nullptr;
    int center_point_box = 0;
    switch (origNode->get_box_encoding()) {
    case ov::opset9::NonMaxSuppression::BoxEncodingType::CENTER:
        center_point_box = 1;
        break;
    case ov::opset9::NonMaxSuppression::BoxEncodingType::CORNER:
        center_point_box = 0;
        break;
    default:
        VPUX_THROW("NonMaxSuppression layer has unsupported box encoding");
    }
    const auto boxEncodingAttr = importBoxEncodingType(center_point_box);
    const auto sortResultDescendingAttr = origNode->get_sort_result_descending() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::NonMaxSuppressionOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                      inputs[3], inputs[4], softNmsSigma, boxEncodingAttr,
                                                      sortResultDescendingAttr, nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& origNode) {
    static_assert(
            std::is_same<std::decay<decltype(*origNode)>::type, ov::op::internal::NonMaxSuppressionIEInternal>::value,
            "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 5) || (inputs.size() == 6),
                      "nGraph NonMaxSuppressionIEInternal node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    const auto softNmsSigma = inputs.size() == 6 ? inputs[5] : nullptr;
    const auto boxEncodingAttr = importBoxEncodingType(origNode->m_center_point_box);
    const auto sortResultDescendingAttr = origNode->m_sort_result_descending ? mlir::UnitAttr::get(_ctx) : nullptr;
    auto op = builder.create<IE::NonMaxSuppressionOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                      inputs[3], inputs[4], softNmsSigma, boxEncodingAttr,
                                                      sortResultDescendingAttr, nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ReverseSequence>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ReverseSequence>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReverseSequence node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto seqAttr = getIntAttr(_ctx, origNode->get_sequence_axis());
    const auto batchAttr = getIntAttr(_ctx, origNode->get_batch_axis());

    auto op = builder.create<IE::ReverseSequenceOp>(createLocation(origNode), inputs[0], inputs[1], seqAttr, batchAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SoftPlus>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::SoftPlus>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph SoftPlus node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SoftPlusOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Less>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Less>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Less node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LessOp>(createLocation(origNode), inputs[0], inputs[1],
                                         importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LessEqual>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LessEqual>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LessEqual node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LessEqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                              importBroadcastType(autob.m_type));

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Greater>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Greater>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Greater node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::GreaterOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GreaterEqual>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::GreaterEqual>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph GreaterEqual node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::GreaterEqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                                 importBroadcastType(autob.m_type));

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::NotEqual>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::NotEqual>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::NotEqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::DepthToSpace>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::DepthToSpace>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto blockSizeAttr = getIntAttr(_ctx, origNode->get_block_size());

    auto op = builder.create<IE::DepthToSpaceOp>(createLocation(origNode), inputs[0], blockSizeAttr,
                                                 importDepthToSpaceMode(origNode->get_mode()));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalNot>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LogicalNot>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::LogicalNotOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalOr>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LogicalOr>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LogicalOr node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LogicalOrOp>(createLocation(origNode), inputs[0], inputs[1],
                                              importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalXor>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LogicalXor>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LogicalXor node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LogicalXorOp>(createLocation(origNode), inputs[0], inputs[1],
                                               importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SpaceToDepth>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::SpaceToDepth>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph SpaceToDepth node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto blockSizeAttr = getIntAttr(_ctx, origNode->get_block_size());
    const auto modeAttr = importSpaceToDepthMode(origNode->get_mode());

    auto op = builder.create<IE::SpaceToDepthOp>(createLocation(origNode), inputs[0], blockSizeAttr, modeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SpaceToBatch>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, opset_latest::SpaceToBatch>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph SpaceToBatch node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SpaceToBatch>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                               nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ExtractImagePatches>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ExtractImagePatches>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1,
                      "nGraph ExtractImagePatches node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto sizesAttr = getIntArrayAttr(_ctx, origNode->get_sizes());
    const auto stridesAttr = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto ratesAttr = getIntArrayAttr(_ctx, origNode->get_rates());
    const auto autoPadAttr = importPadType(origNode->get_auto_pad());

    auto op = builder.create<IE::ExtractImagePatchesOp>(createLocation(origNode), inputs[0], sizesAttr, stridesAttr,
                                                        ratesAttr, autoPadAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Abs>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Abs>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Abs node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AbsOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HSigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::HSigmoid>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph HSigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HSigmoidOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atan>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Atan>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Atan node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AtanOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asin>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Asin>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Asin node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AsinOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acos>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Acos>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Acos node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AcosOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}
void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::PSROIPooling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::PSROIPooling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph PSROIPooling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputDim = getIntAttr(_ctx, origNode->get_output_dim());
    const auto spatialScale = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto groupSize = getIntAttr(_ctx, origNode->get_group_size());
    const auto spatialBinsX = getIntAttr(_ctx, origNode->get_spatial_bins_x());
    const auto spatialBinsY = getIntAttr(_ctx, origNode->get_spatial_bins_y());
    const auto mode = importPSROIPoolingMode(origNode->get_mode());

    auto op = builder.create<IE::PSROIPoolingOp>(createLocation(origNode), inputs[0], inputs[1], outputDim,
                                                 spatialScale, groupSize, spatialBinsX, spatialBinsY, mode);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HardSigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::HardSigmoid>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph HardSigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HardSigmoidOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                                nullptr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::EmbeddingBagOffsetsSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::EmbeddingBagOffsetsSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    IE::EmbeddingBagOffsetsSumOp op;
    mlir::Value defaultIndex = (inputs.size() >= 4) ? inputs[3] : nullptr;
    mlir::Value weights = (inputs.size() >= 5) ? inputs[4] : nullptr;
    op = builder.create<IE::EmbeddingBagOffsetsSumOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                      defaultIndex, weights, nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::EmbeddingSegmentsSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::EmbeddingSegmentsSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4 || inputs.size() == 5 || inputs.size() == 6,
                      "nGraph EmbeddingSegmentsSum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    mlir::Value defaultIndex = (inputs.size() >= 5) ? inputs[4] : nullptr;
    mlir::Value weights = (inputs.size() >= 6) ? inputs[5] : nullptr;
    auto op = builder.create<IE::EmbeddingSegmentsSumOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                         inputs[3], defaultIndex, weights, nullptr, nullptr, nullptr,
                                                         nullptr, nullptr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::EmbeddingBagPackedSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::EmbeddingBagPackedSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                      "nGraph EmbeddingBagPackedSum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    mlir::Value weights = (inputs.size() == 3) ? inputs[2] : nullptr;
    auto op = builder.create<IE::EmbeddingBagPackedSumOp>(createLocation(origNode), inputs[0], inputs[1], weights);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Assign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Assign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Assign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());

    auto op = builder.create<IE::AssignOp>(createLocation(origNode), inputs[0], nameAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::Assign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::Assign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Assign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());

    auto op = builder.create<IE::AssignOp>(createLocation(origNode), inputs[0], nameAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ReadValue>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ReadValue>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ReadValue node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());

    auto op = builder.create<IE::ReadValueOp>(createLocation(origNode), inputs[0], nameAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::ReadValue>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::ReadValue>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ReadValue node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());

    auto op = builder.create<IE::ReadValueOp>(createLocation(origNode), inputs[0], nameAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRUCell>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::GRUCell>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4 || inputs.size() == 5,
                      "nGraph GRUCell node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph GRUCell node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh"}),
                      "nGraph GRUCell node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto hiddenSizeAttr = getIntAttr(_ctx, origNode->get_hidden_size());
    const auto clipAttr = getFPAttr(_ctx, origNode->get_clip());
    const auto shouldLinearBeforeResetAttr = origNode->get_linear_before_reset() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto biasesInput = inputs.size() == 5 ? inputs[4] : nullptr;
    auto op = builder.create<IE::GRUCellOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                            biasesInput, hiddenSizeAttr, shouldLinearBeforeResetAttr, clipAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRUSequence>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::GRUSequence>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 6, "nGraph GRUSequence node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph GRUSequence node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh"}),
                      "nGraph GRUSequence node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto seqLenConstant = dynamic_cast<opset_latest::Constant*>(origNode->input_value(2).get_node());
    VPUX_THROW_UNLESS(seqLenConstant != nullptr,
                      "nGraph GRUSequence node '{0}' has unsupported sequenceLengths input. It must be a Constant node",
                      origNode->get_friendly_name());
    const auto seqLenValues = seqLenConstant->cast_vector<uint32_t>();
    VPUX_THROW_UNLESS(seqLenValues.size() > 0,
                      "nGraph GRUSequence node '{0}' has unsupported sequenceLengths input. It must contain more than "
                      "0 elements",
                      origNode->get_friendly_name());
    const auto isAllLensSame =
            std::all_of(seqLenValues.cbegin(), seqLenValues.cend(), [&seqLenValues](const auto item) {
                return seqLenValues[0] == item;
            });
    VPUX_THROW_UNLESS(
            isAllLensSame,
            "nGraph GRUSequence node '{0}' has unsupported sequenceLengths input. It must contain all the same values",
            origNode->get_friendly_name());
    const auto seqLenAttr = getIntAttr(_ctx, checked_cast<uint32_t>(seqLenValues[0]));

    VPUX_THROW_UNLESS(origNode->get_direction() == ov::op::RecurrentSequenceDirection::FORWARD ||
                              origNode->get_direction() == ov::op::RecurrentSequenceDirection::REVERSE,
                      "nGraph GRUSequence node '{0}' supports direction 'FORWARD' and 'REVERSE'",
                      origNode->get_friendly_name());
    const auto directionAttr = importRNNSequenceDirection(origNode->get_direction());

    const auto hiddenSizeAttr = getIntAttr(_ctx, origNode->get_hidden_size());
    const auto clipAttr = getFPAttr(_ctx, origNode->get_clip());
    const auto shouldLinearBeforeResetAttr = origNode->get_linear_before_reset() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::GRUSequenceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[3], inputs[4],
                                                inputs[5], hiddenSizeAttr, seqLenAttr, directionAttr,
                                                shouldLinearBeforeResetAttr, clipAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::DeformablePSROIPooling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::DeformablePSROIPooling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                      "nGraph DeformablePSROIPooling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputDim = getIntAttr(_ctx, origNode->get_output_dim());
    const auto spatialScale = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto groupSize = getIntAttr(_ctx, origNode->get_group_size());
    const auto spatialBinsX = getIntAttr(_ctx, origNode->get_spatial_bins_x());
    const auto spatialBinsY = getIntAttr(_ctx, origNode->get_spatial_bins_y());
    const auto transStd = getFPAttr(_ctx, origNode->get_trans_std());
    const auto partSize = getIntAttr(_ctx, origNode->get_part_size());
    const auto mode = importDeformablePSROIPoolingMode(origNode->get_mode());

    mlir::Value inputTransformations = (inputs.size() == 3) ? inputs[2] : nullptr;
    auto op = builder.create<IE::DeformablePSROIPoolingOp>(createLocation(origNode), inputs[0], inputs[1],
                                                           inputTransformations, outputDim, spatialScale, groupSize,
                                                           spatialBinsX, spatialBinsY, transStd, partSize, mode);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::DFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v7::DFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::DFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                        (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::RDFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::RDFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::RDFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                         (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::IDFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v7::IDFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::IDFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                         (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::IRDFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::IRDFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::IRDFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                          (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::If>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::If>::value,
                  "opset operation mismatch");

    auto* ctx = builder.getContext();
    const auto inputs = getInputs(origNode);
    const auto location = createLocation(origNode);

    const auto& origNodeThenBody = origNode->get_then_body();
    const auto& origNodeElseBody = origNode->get_else_body();
    NGraphImporter importerThen(ctx, origNodeThenBody, false, _log);
    NGraphImporter importerElse(ctx, origNodeElseBody, false, _log);

    SmallVector<mlir::Type> results;
    results.append(importerThen.getRegionResults());

    auto op = builder.create<IE::IfOp>(location, results, inputs);
    addOutputs(origNode, op);

    auto thenBlock = &op.getThenBranch().emplaceBlock();
    auto thenBranchBuilder = mlir::OpBuilder::atBlockBegin(&op.getThenBranch().front(), nullptr);
    importerThen.buildBlockFromRegion(location, thenBranchBuilder, thenBlock);

    auto elseBlock = &op.getElseBranch().emplaceBlock();
    auto elseBranchBuilder = mlir::OpBuilder::atBlockBegin(&op.getElseBranch().front(), nullptr);
    importerElse.buildBlockFromRegion(location, elseBranchBuilder, elseBlock);
}

//
// IR builder helpers
//

SmallVector<mlir::Value> NGraphImporter::getInputs(const OrigNodePtr& node) {
    SmallVector<mlir::Value> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        out.push_back(_importedVals.at(input.get_source_output()));
    }

    return out;
}

void NGraphImporter::addOutputs(const OrigNodePtr& node, mlir::Operation* op) {
    const auto results = op->getOpResults();

    VPUX_THROW_UNLESS(
            results.size() == node->get_output_size(),
            "Mismatch between original Node '{0}' number of outputs '{1}' and created number of outputs '{2}'",
            node->get_friendly_name(), node->get_output_size(), results.size());

    for (const auto& res : results) {
        _importedVals.emplace(node->output(res.getResultNumber()), res);
    }
}

mlir::Location NGraphImporter::createLocation(const OrigNodePtr& node) {
    const auto nameLoc = mlir::NameLoc::get(mlir::StringAttr::get(_ctx, node->get_friendly_name()));

    SmallVector<mlir::NamedAttribute> fields;
    fields.emplace_back(mlir::StringAttr::get(_ctx, "type"), mlir::StringAttr::get(_ctx, node->get_type_name()));
    fields.emplace_back(mlir::StringAttr::get(_ctx, "name"), mlir::StringAttr::get(_ctx, node->get_friendly_name()));
    auto metadata = mlir::DictionaryAttr::get(_ctx, fields);

    return mlir::FusedLoc::get(_ctx, {nameLoc}, metadata);
}

//
// nGraph attributes importers
//

SmallVector<int64_t> NGraphImporter::importShape(const ov::PartialShape& shape) {
    VPUX_THROW_UNLESS(shape.rank().is_static(), "Dynamically ranked tensors are not supported");

    SmallVector<int64_t> out(checked_cast<size_t>(shape.rank().get_length()));
    for (const auto ind : irange(out.size())) {
        const auto& dim = shape[ind];
        out[ind] = dim.is_static() ? dim.get_length() : mlir::ShapedType::kDynamic;
    }

    return out;
}

mlir::RankedTensorType NGraphImporter::importTensor(const ov::PartialShape& shape, const ov::element::Type& elemType) {
    return mlir::RankedTensorType::get(ArrayRef(importShape(shape)), importPrecision(_ctx, elemType));
}

IE::AutoBroadcastTypeAttr NGraphImporter::importBroadcastType(ov::op::AutoBroadcastType bType) {
    switch (bType) {
    case ov::op::AutoBroadcastType::NONE:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    case ov::op::AutoBroadcastType::NUMPY:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::NUMPY);
    case ov::op::AutoBroadcastType::PDPD:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::PDPD);
    default:
        VPUX_THROW("Unknown AutoBroadcastType");
    }
}

IE::BroadcastTypeAttr NGraphImporter::importBroadcastMode(ov::op::BroadcastType bType) {
    switch (bType) {
    case ov::op::BroadcastType::NUMPY:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::NUMPY);
    case ov::op::BroadcastType::EXPLICIT:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::EXPLICIT);
    case ov::op::BroadcastType::BIDIRECTIONAL:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::BIDIRECTIONAL);
    default:
        VPUX_THROW("Unknown BroadcastMode");
    }
}

IE::RoundingTypeAttr NGraphImporter::importRoundingType(ov::op::RoundingType roundingType) {
    switch (roundingType) {
    case ov::op::RoundingType::FLOOR:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::FLOOR);
    case ov::op::RoundingType::CEIL:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::CEIL);
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}

IE::EpsModeAttr NGraphImporter::importEpsMode(ov::op::EpsMode val) {
    switch (val) {
    case ov::op::EpsMode::ADD:
        return IE::EpsModeAttr::get(_ctx, IE::EpsMode::ADD);
    case ov::op::EpsMode::MAX:
        return IE::EpsModeAttr::get(_ctx, IE::EpsMode::MAX);
    default:
        VPUX_THROW("Unknown EpsMode");
    }
}

IE::MvnEpsModeAttr NGraphImporter::importMvnEpsMode(ov::op::MVNEpsMode val) {
    switch (val) {
    case ov::op::MVNEpsMode::INSIDE_SQRT:
        return IE::MvnEpsModeAttr::get(_ctx, IE::MvnEpsMode::INSIDE_SQRT);
    case ov::op::MVNEpsMode::OUTSIDE_SQRT:
        return IE::MvnEpsModeAttr::get(_ctx, IE::MvnEpsMode::OUTSIDE_SQRT);
    default:
        VPUX_THROW("Unknown MvnEpsMode");
    }
}

IE::TopKModeAttr NGraphImporter::importTopKMode(ov::op::TopKMode val) {
    switch (val) {
    case ov::op::TopKMode::MAX:
        return IE::TopKModeAttr::get(_ctx, IE::TopKMode::MAX);
    case ov::op::TopKMode::MIN:
        return IE::TopKModeAttr::get(_ctx, IE::TopKMode::MIN);
    default:
        VPUX_THROW("Unknown TopKMode");
    }
}

IE::TopKSortTypeAttr NGraphImporter::importTopKSortType(ov::op::TopKSortType val) {
    switch (val) {
    case ov::op::TopKSortType::NONE:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::NONE);
    case ov::op::TopKSortType::SORT_INDICES:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::SORT_INDICES);
    case ov::op::TopKSortType::SORT_VALUES:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::SORT_VALUES);
    default:
        VPUX_THROW("Unknown TopKSortType");
    }
}

IE::GridSampleModeAttr NGraphImporter::importGridSampleMode(const ov::op::v9::GridSample::InterpolationMode& val) {
    IE::GridSampleModeAttr attr;
    if (val == ov::op::v9::GridSample::InterpolationMode::BILINEAR) {
        attr = IE::GridSampleModeAttr::get(_ctx, IE::GridSampleMode::BILINEAR);
    } else if (val == ov::op::v9::GridSample::InterpolationMode::BICUBIC) {
        attr = IE::GridSampleModeAttr::get(_ctx, IE::GridSampleMode::BICUBIC);
    } else if (val == ov::op::v9::GridSample::InterpolationMode::NEAREST) {
        attr = IE::GridSampleModeAttr::get(_ctx, IE::GridSampleMode::NEAREST);
    } else {
        VPUX_THROW("Unknown GridSampleMode");
    }
    return attr;
}

IE::GridSamplePaddingModeAttr NGraphImporter::importGridSamplePaddingMode(
        const ov::op::v9::GridSample::PaddingMode& val) {
    IE::GridSamplePaddingModeAttr attr;
    if (val == ov::op::v9::GridSample::PaddingMode::ZEROS) {
        attr = IE::GridSamplePaddingModeAttr::get(_ctx, IE::GridSamplePaddingMode::ZEROS);
    } else if (val == ov::op::v9::GridSample::PaddingMode::BORDER) {
        attr = IE::GridSamplePaddingModeAttr::get(_ctx, IE::GridSamplePaddingMode::BORDER);
    } else if (val == ov::op::v9::GridSample::PaddingMode::REFLECTION) {
        attr = IE::GridSamplePaddingModeAttr::get(_ctx, IE::GridSamplePaddingMode::REFLECTION);
    } else {
        VPUX_THROW("Unknown GridSamplePaddingMode");
    }
    return attr;
}

IE::ProposalAttr NGraphImporter::importProposalAttrs(const ov::op::v0::Proposal::Attributes& val) {
    const auto baseSizeAttr = getIntAttr(_ctx, val.base_size);
    const auto preNmsTopNAttr = getIntAttr(_ctx, val.pre_nms_topn);
    const auto postNmsTopNAttr = getIntAttr(_ctx, val.post_nms_topn);
    const auto nmsThreshNAttr = getFPAttr(_ctx, val.nms_thresh);
    const auto featStrideAttr = getIntAttr(_ctx, val.feat_stride);
    const auto minSizeNAttr = getIntAttr(_ctx, val.min_size);
    const auto ratioAttr = getFPArrayAttr(_ctx, val.ratio);
    const auto scaleAttr = getFPArrayAttr(_ctx, val.scale);
    const auto clipBeforeNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_before_nms);
    const auto clipAfterNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_after_nms);
    const auto normalizeAttr = mlir::BoolAttr::get(_ctx, val.normalize);
    const auto boxSizeScaleAttr = getFPAttr(_ctx, val.box_size_scale);
    const auto boxCoordinateScaleAttr = getFPAttr(_ctx, val.box_coordinate_scale);
    const auto frameworkAttr = mlir::StringAttr::get(_ctx, val.framework);
    const auto inferProbsAttr = mlir::BoolAttr::get(_ctx, val.infer_probs);

    return IE::ProposalAttr::get(_ctx, baseSizeAttr, preNmsTopNAttr, postNmsTopNAttr, nmsThreshNAttr, featStrideAttr,
                                 minSizeNAttr, ratioAttr, scaleAttr, clipBeforeNmsAttr, clipAfterNmsAttr, normalizeAttr,
                                 boxSizeScaleAttr, boxCoordinateScaleAttr, frameworkAttr, inferProbsAttr);
}

IE::InterpolateAttr NGraphImporter::importInterpolateAttrs(const opset_latest::Interpolate::InterpolateAttrs& val) {
    // mode
    IE::InterpolateModeAttr modeAttr;
    switch (val.mode) {
    case opset_latest::Interpolate::InterpolateMode::NEAREST:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::NEAREST);
        break;
    case opset_latest::Interpolate::InterpolateMode::LINEAR:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::LINEAR);
        break;
    case opset_latest::Interpolate::InterpolateMode::LINEAR_ONNX:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::LINEAR_ONNX);
        break;
    case opset_latest::Interpolate::InterpolateMode::CUBIC:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::CUBIC);
        break;
    default:
        VPUX_THROW("Unsupported interpolate mode");
    }

    // shape calculation mode
    IE::InterpolateCalcModeAttr calcModeAttr;
    switch (val.shape_calculation_mode) {
    case opset_latest::Interpolate::ShapeCalcMode::SIZES:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::SIZES);
        break;
    case opset_latest::Interpolate::ShapeCalcMode::SCALES:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::SCALES);
        break;
    default:
        VPUX_THROW("Unsupported interpolate shape calculation mode");
    }

    // coordinate transformation mode
    IE::InterpolateCoordModeAttr coordModeAttr;
    switch (val.coordinate_transformation_mode) {
    case opset_latest::Interpolate::CoordinateTransformMode::HALF_PIXEL:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::HALF_PIXEL);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::ASYMMETRIC:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::ASYMMETRIC);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::ALIGN_CORNERS:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::ALIGN_CORNERS);
        break;
    default:
        VPUX_THROW("Unsupported interpolate coordinate transformation mode");
    }

    // coordinate transformation mode
    IE::InterpolateNearestModeAttr nearestModeAttr;
    switch (val.nearest_mode) {
    case opset_latest::Interpolate::NearestMode::ROUND_PREFER_FLOOR:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::ROUND_PREFER_FLOOR);
        break;
    case opset_latest::Interpolate::NearestMode::ROUND_PREFER_CEIL:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::ROUND_PREFER_CEIL);
        break;
    case opset_latest::Interpolate::NearestMode::FLOOR:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::FLOOR);
        break;
    case opset_latest::Interpolate::NearestMode::CEIL:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::CEIL);
        break;
    case opset_latest::Interpolate::NearestMode::SIMPLE:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::SIMPLE);
        break;
    default:
        VPUX_THROW("Unsupported interpolate nearest mode");
    }

    const auto antialiasAttr = mlir::BoolAttr::get(_ctx, val.antialias);
    const auto padsBeginAttr = getIntArrayAttr(_ctx, val.pads_begin);
    const auto padsEndAttr = getIntArrayAttr(_ctx, val.pads_end);
    const auto cubeCoeffAttr = getFPAttr(_ctx, val.cube_coeff);

    return IE::InterpolateAttr::get(_ctx, modeAttr, calcModeAttr, coordModeAttr, nearestModeAttr, antialiasAttr,
                                    padsBeginAttr, padsEndAttr, cubeCoeffAttr);
}

IE::DetectionOutputCodeTypeAttr NGraphImporter::importDetectionOutputCodeType(const std::string& codeType) {
    if (codeType == "caffe.PriorBoxParameter.CENTER_SIZE") {
        return IE::DetectionOutputCodeTypeAttr::get(_ctx, IE::DetectionOutputCodeType::CENTER_SIZE);
    } else if (codeType == "caffe.PriorBoxParameter.CORNER") {
        return IE::DetectionOutputCodeTypeAttr::get(_ctx, IE::DetectionOutputCodeType::CORNER);
    } else if (codeType == "caffe.PriorBoxParameter.CORNER_SIZE") {
        return IE::DetectionOutputCodeTypeAttr::get(_ctx, IE::DetectionOutputCodeType::CORNER_SIZE);
    }

    VPUX_THROW("Unknown DetectionOutput code_type");
}

IE::DetectionOutputAttr NGraphImporter::importDetectionOutputAttrs(const ov::op::v0::DetectionOutput::Attributes& val) {
    const auto numClassesAttr = getIntAttr(_ctx, val.num_classes);
    const auto backgroundLabelIdAttr = getIntAttr(_ctx, val.background_label_id);
    const auto topKAttr = getIntAttr(_ctx, val.top_k);

    const auto varianceEncodedInTargetAttr = mlir::BoolAttr::get(_ctx, val.variance_encoded_in_target);

    const auto keepTopKAttr = getIntArrayAttr(_ctx, val.keep_top_k);
    const auto codeTypeAttr = importDetectionOutputCodeType(val.code_type);

    const auto shareLocationAttr = mlir::BoolAttr::get(_ctx, val.share_location);

    const auto nmsThresholdAttr = getFPAttr(_ctx, val.nms_threshold);
    const auto confidenceThresholdAttr = getFPAttr(_ctx, val.confidence_threshold);

    const auto clipAfterNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_after_nms);
    const auto clipBeforeNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_before_nms);
    const auto decreaseLabel_idAttr = mlir::BoolAttr::get(_ctx, val.decrease_label_id);
    const auto normalizedAttr = mlir::BoolAttr::get(_ctx, val.normalized);

    const auto inputHeightAttr = getIntAttr(_ctx, val.input_height);
    const auto inputWidthAttr = getIntAttr(_ctx, val.input_width);

    const auto objectnessScoreAttr = getFPAttr(_ctx, val.objectness_score);

    return IE::DetectionOutputAttr::get(_ctx, numClassesAttr, backgroundLabelIdAttr, topKAttr,
                                        varianceEncodedInTargetAttr, keepTopKAttr, codeTypeAttr, shareLocationAttr,
                                        nmsThresholdAttr, confidenceThresholdAttr, clipAfterNmsAttr, clipBeforeNmsAttr,
                                        decreaseLabel_idAttr, normalizedAttr, inputHeightAttr, inputWidthAttr,
                                        objectnessScoreAttr);
}

IE::ROIPoolingMethodAttr NGraphImporter::importROIPoolingMethod(const std::string& method) {
    IE::ROIPoolingMethodAttr attr;
    if (method == "max") {
        attr = IE::ROIPoolingMethodAttr::get(_ctx, IE::ROIPoolingMethod::MAX);
    } else if (method == "bilinear") {
        attr = IE::ROIPoolingMethodAttr::get(_ctx, IE::ROIPoolingMethod::BILINEAR);
    } else {
        VPUX_THROW("Unknown ROIPoolingMethod");
    }
    return attr;
}

IE::PSROIPoolingModeAttr NGraphImporter::importPSROIPoolingMode(const std::string& mode) {
    if (mode == "average") {
        return IE::PSROIPoolingModeAttr::get(_ctx, IE::PSROIPoolingMode::AVERAGE);
    } else if (mode == "bilinear") {
        return IE::PSROIPoolingModeAttr::get(_ctx, IE::PSROIPoolingMode::BILINEAR);
    }

    VPUX_THROW("Unknown PSROIPoolingMode: {0}", mode);
}

IE::ROIAlignMethodAttr NGraphImporter::importROIAlignMethod(const ov::op::v9::ROIAlign::PoolingMode& mode) {
    IE::ROIAlignMethodAttr attr;
    if (mode == ov::op::v9::ROIAlign::PoolingMode::AVG) {
        attr = IE::ROIAlignMethodAttr::get(_ctx, IE::ROIAlignMethod::AVG);
    } else if (mode == ov::op::v9::ROIAlign::PoolingMode::MAX) {
        attr = IE::ROIAlignMethodAttr::get(_ctx, IE::ROIAlignMethod::MAX);
    } else {
        VPUX_THROW("Unknown ROIAlignMethod");
    }
    return attr;
}

IE::ROIAlignAlignedMethodAttr NGraphImporter::importROIAlignAlignedMethod(
        const ov::op::v9::ROIAlign::AlignedMode& mode) {
    IE::ROIAlignAlignedMethodAttr attr;
    if (mode == ov::op::v9::ROIAlign::AlignedMode::ASYMMETRIC) {
        attr = IE::ROIAlignAlignedMethodAttr::get(_ctx, IE::ROIAlignAlignedMethod::ASYMMETRIC);
    } else if (mode == ov::op::v9::ROIAlign::AlignedMode::HALF_PIXEL) {
        attr = IE::ROIAlignAlignedMethodAttr::get(_ctx, IE::ROIAlignAlignedMethod::HALF_PIXEL);
    } else if (mode == ov::op::v9::ROIAlign::AlignedMode::HALF_PIXEL_FOR_NN) {
        attr = IE::ROIAlignAlignedMethodAttr::get(_ctx, IE::ROIAlignAlignedMethod::HALF_PIXEL_FOR_NN);
    } else {
        VPUX_THROW("Unknown ROIAlignAlignedMethod");
    }
    return attr;
}

IE::RNNSequenceDirectionAttr NGraphImporter::importRNNSequenceDirection(const ov::op::RecurrentSequenceDirection val) {
    IE::RNNSequenceDirectionAttr attr;
    if (val == ov::op::RecurrentSequenceDirection::FORWARD) {
        attr = IE::RNNSequenceDirectionAttr::get(_ctx, IE::RNNSequenceDirection::FORWARD);
    } else if (val == ov::op::RecurrentSequenceDirection::REVERSE) {
        attr = IE::RNNSequenceDirectionAttr::get(_ctx, IE::RNNSequenceDirection::REVERSE);
    } else {
        VPUX_THROW("Unknown RNNSequence direction");
    }
    return attr;
}

IE::BoxEncodingTypeAttr NGraphImporter::importBoxEncodingType(const int box_encoding_type) {
    return box_encoding_type ? IE::BoxEncodingTypeAttr::get(_ctx, IE::BoxEncodingType::CENTER)
                             : IE::BoxEncodingTypeAttr::get(_ctx, IE::BoxEncodingType::CORNER);
}

IE::PadModeAttr NGraphImporter::importPadMode(const ov::op::PadMode val) {
    IE::PadModeAttr attr;
    switch (val) {
    case ov::op::PadMode::CONSTANT:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::CONSTANT);
        break;
    case ov::op::PadMode::EDGE:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::EDGE);
        break;
    case ov::op::PadMode::REFLECT:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::REFLECT);
        break;
    case ov::op::PadMode::SYMMETRIC:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::SYMMETRIC);
        break;
    default:
        VPUX_THROW("Unknown PadMode");
    }
    return attr;
}

IE::RoundModeAttr NGraphImporter::importRoundMode(const ov::op::v5::Round::RoundMode val) {
    IE::RoundModeAttr attr;
    switch (val) {
    case ov::op::v5::Round::RoundMode::HALF_TO_EVEN:
        attr = IE::RoundModeAttr::get(_ctx, IE::RoundMode::HALF_TO_EVEN);
        break;
    case ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO:
        attr = IE::RoundModeAttr::get(_ctx, IE::RoundMode::HALF_AWAY_FROM_ZERO);
        break;
    default:
        VPUX_THROW("Unknown RoundMode");
    }
    return attr;
}

IE::DepthToSpaceModeAttr NGraphImporter::importDepthToSpaceMode(const ov::op::v0::DepthToSpace::DepthToSpaceMode val) {
    IE::DepthToSpaceModeAttr attr;
    switch (val) {
    case ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
        attr = IE::DepthToSpaceModeAttr::get(_ctx, IE::DepthToSpaceMode::BLOCKS_FIRST);
        break;
    case ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
        attr = IE::DepthToSpaceModeAttr::get(_ctx, IE::DepthToSpaceMode::DEPTH_FIRST);
        break;
    default:
        VPUX_THROW("Unknown DepthToSpace Mode");
    }
    return attr;
}

IE::SpaceToDepthModeAttr NGraphImporter::importSpaceToDepthMode(const ov::op::v0::SpaceToDepth::SpaceToDepthMode val) {
    IE::SpaceToDepthModeAttr attr;
    switch (val) {
    case ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
        attr = IE::SpaceToDepthModeAttr::get(_ctx, IE::SpaceToDepthMode::BLOCKS_FIRST);
        break;
    case ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
        attr = IE::SpaceToDepthModeAttr::get(_ctx, IE::SpaceToDepthMode::DEPTH_FIRST);
        break;
    default:
        VPUX_THROW("Unknown SpaceToDepthMode");
    }
    return attr;
}

IE::PadTypeAttr NGraphImporter::importPadType(ov::op::PadType padding) {
    switch (padding) {
    case ov::op::PadType::SAME_UPPER:
        return IE::PadTypeAttr::get(_ctx, IE::PadType::SAME_UPPER);
    case ov::op::PadType::SAME_LOWER:
        return IE::PadTypeAttr::get(_ctx, IE::PadType::SAME_LOWER);
    case ov::op::PadType::VALID:
        return IE::PadTypeAttr::get(_ctx, IE::PadType::VALID);
    default:
        VPUX_THROW("Unknown PadType {0}", static_cast<int>(padding));
    }
}

IE::DeformablePSROIPoolingModeAttr NGraphImporter::importDeformablePSROIPoolingMode(const std::string& mode) {
    if (mode == "average") {
        return IE::DeformablePSROIPoolingModeAttr::get(_ctx, IE::DeformablePSROIPoolingMode::AVERAGE);
    } else if (mode == "bilinear_deformable") {
        return IE::DeformablePSROIPoolingModeAttr::get(_ctx, IE::DeformablePSROIPoolingMode::BILINEAR_DEFORMABLE);
    }

    VPUX_THROW("Unknown DeformablePSROIPoolingMode: {0}", mode);
}

mlir::RankedTensorType importUserTensor(mlir::MLIRContext* ctx, const ov::descriptor::Tensor& tensor) {
    const Shape shape(tensor.get_shape().begin(), tensor.get_shape().end());
    const auto precision = importPrecision(ctx, tensor.get_element_type());
    return getTensorType(shape, precision, DimsOrder::fromNumDims(tensor.get_shape().size()), nullptr);
}

//
// runNGraphPasses
//

static void addCommonOptimizationsPasses(ov::pass::Manager& manager) {
    // Disable low_precision_enabled as all plugins handle low-precision sub-graph manually
    // before CommonOptimization pipeline execution
    manager.register_pass<ov::pass::MOCTransformations>(true, false);

    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::PadFusionConvolution>();
    pass_config->disable<ov::pass::PadFusionGroupConvolution>();
    pass_config->disable<ov::pass::MVNFusionWithConstantsInside>();
    pass_config->disable<ov::pass::PullThroughReduce>();
    pass_config->disable<ov::pass::AddFakeQuantizeFusion>();
    pass_config->disable<ov::pass::FakeQuantizeMulFusion>();

    // NMS conversion passes
    manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();

    auto static_shape = manager.register_pass<ov::pass::GraphRewrite>();
    static_shape->add_matcher<ov::pass::ConvertNMS9ToNMSIEInternal>();
    static_shape->set_name("ov::pass::CommonStaticShape");

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    common_fusions->add_matcher<ov::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ov::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ov::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::TransposeToReshape>();
    common_fusions->set_name("ov::pass::CommonFusions");

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::Gelu7Downgrade>();
    decomp->add_matcher<ov::pass::BidirectionalSequenceDecomposition>();
    decomp->add_matcher<ov::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ov::pass::BatchNormDecomposition>();
    decomp->add_matcher<ov::pass::EinsumDecomposition>();
    decomp->add_matcher<ov::pass::DropoutWithRandomUniformReplacer>();
    decomp->set_name("ov::pass::CommonDecompositions");

    // CF is required after all decompositions
    manager.register_pass<ov::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ov::pass::LinOpSequenceFusion>();
    manager.register_pass<ov::pass::UnrollIf>();
    manager.register_pass<ov::pass::UnrollTensorIterator>();

    auto conv_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    conv_fusions->add_matcher<ov::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionBackpropDataFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    conv_fusions->set_name("ov::pass::ConvFusions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertGather1ToGather7>();
    manager.register_pass<ov::pass::ConvertGather7ToGather8>();
    manager.register_pass<ov::pass::ConvertDeformableConv8To1>();
    manager.register_pass<ov::pass::ConvertMaxPool8ToMaxPool1>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.register_pass<ov::pass::ConvertDetectionOutput8ToDetectionOutput1>();

    // StridesOptimization should be at the very end
    // because we cannot insert any MaxPools since they may prevent
    // other optimizations
    manager.register_pass<ov::pass::StridesOptimization>();
}

void NGraphPasses::runNGraphPasses(const std::shared_ptr<ov::Model>& netGraph, mlir::TimingScope& rootTiming,
                                   const vpux::VPU::ArchKind arch) {
    auto scopeTiming = rootTiming.nest("Common nGraph passes");

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<vpux::passes::ConvertInstanceNormToMVN>();
    manager.register_pass<vpux::pass::RemoveSplitConcat>();
    ov::element::TypeVector decompression_precisions{
            ov::element::u4, ov::element::i4, ov::element::nf4, ov::element::u8, ov::element::i8,
    };
    manager.register_pass<ov::pass::MarkDequantizationSubgraph>(decompression_precisions, /*fold_subtract_const=*/true);
    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<vpux::pass::FuseScaleShift>();
    manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<vpux::passes::OnnxReorgPatternToDarkNetReorg>();
    manager.register_pass<vpux::pass::FuseScaleAfterClamp>();
    addCommonOptimizationsPasses(manager);

    manager.register_pass<vpux::passes::PropagateFQ>();
    // Disables for VPUX37XX
    if (!supportsPerInputEltwiseScale(arch)) {
        manager.register_pass<vpux::passes::AlignScales>();
    }

    // we need additionally propagate FQs because some ReLUs may be removed
    manager.register_pass<vpux::passes::PropagateFQ>();
    manager.register_pass<vpux::passes::CleanUpFQ>();

    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
    manager.register_pass<ov::pass::UnrollTensorIterator>();

    // MVN Conversion passes
    manager.register_pass<vpux::passes::ConvertLayerNormToMVN>();
    manager.register_pass<vpux::passes::ConvertMVN6toMVN1>();

    manager.register_pass<vpux::passes::ConvertVariadicSplitToStridedSliceOp>();

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (const auto serializeCanonicalModel = std::getenv("NPU_SERIALIZE_CANONICAL_MODEL")) {
        if (vpux::envVarStrToBool("NPU_SERIALIZE_CANONICAL_MODEL", serializeCanonicalModel)) {
            const std::string graphName = netGraph->get_friendly_name();
            manager.register_pass<ov::pass::Serialize>(graphName + "_canonical.xml", graphName + "_canonical.bin");
        }
    }
#endif

    manager.run_passes(netGraph);
}

//
// addCNNNetworkOp
//

void addCNNNetworkOp(mlir::OpBuilder& builder, mlir::FlatSymbolRefAttr mainFuncName,
                     const std::shared_ptr<ov::Model>& model, mlir::TimingScope& rootTiming, bool enableProfiling) {
    auto scopeTiming = rootTiming.nest("Add CNNNetwork Operation");

    const auto parameters = model->get_parameters();
    const auto results = model->get_results();

    auto* ctx = builder.getContext();

    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(ctx), mainFuncName, enableProfiling);
    cnnOp.getInputsInfo().emplaceBlock();
    cnnOp.getOutputsInfo().emplaceBlock();
    if (enableProfiling) {
        cnnOp.getProfilingOutputsInfo().front().emplaceBlock();
    }

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.getInputsInfo().front(), builder.getListener());
    for (const auto& parameter : parameters) {
        const auto& parameterName = parameter->get_friendly_name();

        const auto nameAttr = mlir::StringAttr::get(ctx, parameterName);
        const auto userTypeAttr = mlir::TypeAttr::get(importUserTensor(ctx, parameter->get_output_tensor(0)));

        inputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr,
                                                 /*profilingSectionsCount=*/0);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.getOutputsInfo().front(), builder.getListener());
    for (const auto& result : results) {
        const auto resultName = ov::op::util::get_ie_output_name(result->input_value(0));

        const auto nameAttr = mlir::StringAttr::get(ctx, resultName);
        const auto userTypeAttr = mlir::TypeAttr::get(importUserTensor(ctx, result->get_output_tensor(0)));

        outputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr,
                                                  /*profilingSectionsCount=*/0);
    }
}

//
// addSparsityStatistics
//

void addSparsityStatistics(const std::shared_ptr<ov::Model>& model, mlir::ModuleOp module, Logger log) {
    using RTMap = std::map<std::string, ov::Any>;

    auto maybeRt = model->get_rt_info();
    if (maybeRt.empty()) {
        log.trace("Missed RT info");
        return;
    }
    log.trace("Found RT info");
    if (!model->has_rt_info(NGRAPH_ACT_SPARSITY_STATS_KEY)) {
        return;
    }
    log.trace("Found activation statistics");
    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    auto* ctx = builder.getContext();

    auto statsOp = builder.create<IE::SparsityStatisticsOp>(mlir::UnknownLoc::get(ctx));
    statsOp.getSparsityInfo().emplaceBlock();

    auto statsBuilder = mlir::OpBuilder::atBlockBegin(&statsOp.getSparsityInfo().front(), builder.getListener());

    auto actSparsityStats = model->get_rt_info<RTMap>(NGRAPH_ACT_SPARSITY_STATS_KEY);
    for (const auto& x : actSparsityStats) {
        const auto key = x.first;
        const auto nodeName = getSparsityStatsFieldChecked<std::string>(model, key, "node_name");
        const auto portId = getSparsityStatsFieldChecked<int>(model, key, "port_id");
        const auto ratio = getSparsityStatsFieldChecked<double>(model, key, "statistic");

        const auto nameAttr = mlir::StringAttr::get(ctx, nodeName);
        const auto inputAttr = getIntAttr(ctx, portId);
        const auto ratioAttr = getFPAttr(ctx, ratio);
        statsBuilder.create<IE::SparsityInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, inputAttr, ratioAttr);
    }
}

//
// importNetwork
//

mlir::OwningOpRef<mlir::ModuleOp> vpux::IE::importNetwork(mlir::MLIRContext* ctx,
                                                          const std::shared_ptr<ov::Model>& model, bool sharedConstants,
                                                          mlir::TimingScope& rootTiming, bool enableProfiling,
                                                          bool stubLayers, vpux::VPU::ArchKind arch, Logger log) {
    log.setName("IE::FrontEnd::importNetwork");

    log.trace("Load IE::FrontEnd dependent Dialects");
    ctx->loadDialect<IE::IEDialect>();

    log.trace("Run common nGraph passes");
    NGraphPasses::runNGraphPasses(model, rootTiming, arch);

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef(model->get_friendly_name()));
    addSparsityStatistics(model, module, log);
    const auto mainFuncName = mlir::FlatSymbolRefAttr::get(ctx, "main");

    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    log.trace("Add CNNNetwork Operation");
    addCNNNetworkOp(builder, mainFuncName, model, rootTiming, enableProfiling);

    log.trace("Import nGraph function");
    NGraphImporter importer(ctx, model, sharedConstants, log);
    importer.buildMainFunc(builder, mainFuncName.getValue(), rootTiming, stubLayers);

    log.trace("Validate MLIR module");
    auto finalTiming = rootTiming.nest("Validate MLIR module");
    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)), "Failed to create a valid MLIR module for the IR model");

    return module;
}
