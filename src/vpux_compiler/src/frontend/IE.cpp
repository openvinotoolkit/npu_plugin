//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/frontend/IE.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/IE/hash.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>

#include <ie_common.h>
#include <ie_layouts.h>
#include <ie_precision.hpp>

#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/op_conversions/convert_minimum_to_power_and_max.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_negative.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>

using namespace vpux;

namespace {

class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ngraph::Function> netGraph, bool sharedConstants,
                   Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

public:
    mlir::FuncOp buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName);

private:
    using OrigNode = ngraph::Node;
    using OrigNodePtr = std::shared_ptr<OrigNode>;
    using NodeOutputMap = std::unordered_map<ngraph::Output<OrigNode>, mlir::Value>;

private:
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Constant>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Convert>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Softmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Tile>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Relu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Split>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Power>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Multiply>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Convolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::GroupConvolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::ConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::AvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::MaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::PriorBox>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::PriorBoxClustered>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Gather>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Clamp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Elu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Reshape>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Squeeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Sigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::LRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Unsqueeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Minimum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Maximum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Add>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Divide>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::SquaredDifference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::FloorMod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Proposal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::FakeQuantize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::MatMul>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Tanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Exp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::HSwish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Transpose>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Interpolate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::RegionYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset2::ReorgYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::DetectionOutput>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::NormalizeL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Concat>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset2::ROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::StridedSlice>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::PRelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::Swish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::GRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Negative>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::CTCGreedyDecoder>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset6::CTCGreedyDecoderSeqLen>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset6::Pad>& origNode);

    template <class NodeType>
    void parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
        parseNode(builder, std::dynamic_pointer_cast<NodeType>(origNode));
    }

    void parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
    }

private:
    SmallVector<mlir::Value> getInputs(const OrigNodePtr& node);
    void addOutputs(const OrigNodePtr& node, mlir::Operation* op);
    mlir::Location createLocation(const OrigNodePtr& node);

private:
    static SmallVector<int64_t> importShape(const ngraph::PartialShape& shape);
    mlir::Type importElemType(const ngraph::element::Type& elemType);
    mlir::RankedTensorType importTensor(const ngraph::PartialShape& shape, const ngraph::element::Type& elemType);
    IE::AutoBroadcastTypeAttr importBroadcastType(ngraph::op::AutoBroadcastType bType);
    IE::RoundingTypeAttr importRoundingType(ngraph::op::RoundingType roundingType);
    IE::EpsModeAttr importEpsMode(ngraph::op::EpsMode val);
    IE::TopKModeAttr importTopKMode(ngraph::op::TopKMode val);
    IE::TopKSortTypeAttr importTopKSortType(ngraph::op::TopKSortType val);
    IE::ProposalAttr importProposalAttrs(const ngraph::op::ProposalAttrs& val);
    IE::InterpolateAttr importInterpolateAttrs(const ngraph::op::InterpolateAttrs& val);
    IE::DetectionOutputAttr importDetectionOutputAttrs(const ngraph::op::DetectionOutputAttrs& val);
    IE::ROIPoolingMethodAttr importROIPoolingMethod(const std::string& method);
    IE::PadModeAttr importPadMode(const ngraph::op::PadMode val);

private:
    mlir::MLIRContext* _ctx = nullptr;
    std::shared_ptr<const ngraph::Function> _netGraph;
    bool _sharedConstants = false;
    Logger _log;

    NodeOutputMap _importedVals;
};

//
// buildMainFunc
//

mlir::FuncOp NGraphImporter::buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName) {
    using Callback = void (NGraphImporter::*)(mlir::OpBuilder & builder, const OrigNodePtr& origNode);
    using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

#define MAP_ENTRY(_NodeType_) \
    { _NodeType_::type_info, &NGraphImporter::parseDispatch<_NodeType_> }

    static const DispatchMap dispatchMap{
            {ngraph::op::Parameter::type_info, &NGraphImporter::parseEmpty},
            {ngraph::op::Result::type_info, &NGraphImporter::parseEmpty},

            MAP_ENTRY(ngraph::opset1::Constant),
            MAP_ENTRY(ngraph::opset1::Convert),
            MAP_ENTRY(ngraph::opset1::Softmax),
            MAP_ENTRY(ngraph::opset1::Tile),
            MAP_ENTRY(ngraph::opset1::Split),
            MAP_ENTRY(ngraph::opset1::Power),
            MAP_ENTRY(ngraph::opset1::Multiply),
            MAP_ENTRY(ngraph::opset1::Relu),
            MAP_ENTRY(ngraph::opset1::Convolution),
            MAP_ENTRY(ngraph::opset1::GroupConvolution),
            MAP_ENTRY(ngraph::opset1::ConvolutionBackpropData),
            MAP_ENTRY(ngraph::opset1::AvgPool),
            MAP_ENTRY(ngraph::opset1::MaxPool),
            MAP_ENTRY(ngraph::opset1::PriorBox),
            MAP_ENTRY(ngraph::opset1::PriorBoxClustered),
            MAP_ENTRY(ngraph::opset1::Gather),
            MAP_ENTRY(ngraph::opset1::Clamp),
            MAP_ENTRY(ngraph::opset1::Elu),
            MAP_ENTRY(ngraph::opset1::Reshape),
            MAP_ENTRY(ngraph::opset1::Squeeze),
            MAP_ENTRY(ngraph::opset1::Sigmoid),
            MAP_ENTRY(ngraph::opset1::LRN),
            MAP_ENTRY(ngraph::opset1::Unsqueeze),
            MAP_ENTRY(ngraph::opset1::Minimum),
            MAP_ENTRY(ngraph::opset1::Maximum),
            MAP_ENTRY(ngraph::opset1::Add),
            MAP_ENTRY(ngraph::opset1::Divide),
            MAP_ENTRY(ngraph::opset1::SquaredDifference),
            MAP_ENTRY(ngraph::opset1::FloorMod),
            MAP_ENTRY(ngraph::opset1::Proposal),
            MAP_ENTRY(ngraph::opset1::FakeQuantize),
            MAP_ENTRY(ngraph::opset1::MatMul),
            MAP_ENTRY(ngraph::opset1::Tanh),
            MAP_ENTRY(ngraph::opset1::Exp),
            MAP_ENTRY(ngraph::opset4::HSwish),
            MAP_ENTRY(ngraph::opset1::Transpose),
            MAP_ENTRY(ngraph::opset1::Interpolate),
            MAP_ENTRY(ngraph::opset1::TopK),
            MAP_ENTRY(ngraph::opset1::RegionYolo),
            MAP_ENTRY(ngraph::opset2::ReorgYolo),
            MAP_ENTRY(ngraph::opset1::DetectionOutput),
            MAP_ENTRY(ngraph::opset1::NormalizeL2),
            MAP_ENTRY(ngraph::opset1::Concat),
            MAP_ENTRY(ngraph::opset2::ROIPooling),
            MAP_ENTRY(ngraph::opset1::StridedSlice),
            MAP_ENTRY(ngraph::opset1::PRelu),
            MAP_ENTRY(ngraph::opset4::Swish),
            MAP_ENTRY(ngraph::opset1::GRN),
            MAP_ENTRY(ngraph::opset1::Negative),
            MAP_ENTRY(ngraph::opset1::CTCGreedyDecoder),
            MAP_ENTRY(ngraph::opset6::CTCGreedyDecoderSeqLen),
            MAP_ENTRY(ngraph::opset6::Pad),
    };

#undef MAP_ENTRY

    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(_netGraph->get_parameters().size());
    for (const auto& param : _netGraph->get_parameters()) {
        inputTypes.push_back(importTensor(param->get_partial_shape(), param->get_element_type()));
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }

    const auto funcType = mlir::FunctionType::get(_ctx, makeArrayRef(inputTypes), makeArrayRef(outputTypes));

    auto func = moduleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(_ctx), funcName, funcType);

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

        const auto dispatchIt = dispatchMap.find(origNode->get_type_info());
        VPUX_THROW_UNLESS(dispatchIt != dispatchMap.end(), "Unsupported operation {0} with type {1}",
                          origNode->get_friendly_name(), origNode->get_type_name());

        const auto parser = dispatchIt->second;
        (this->*parser)(builder, origNode);
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

    builder.create<mlir::ReturnOp>(mlir::UnknownLoc::get(_ctx), makeArrayRef(funcOutputs));

    return func;
}

//
// Parsers
//

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Constant>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.empty(), "nGraph Constant node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto tensorType = importTensor(origNode->get_output_partial_shape(0), origNode->get_output_element_type(0));

    const auto numElems = tensorType.getNumElements();
    const Byte elemTypeSize = getElemTypeSize(tensorType);

    mlir::ElementsAttr value;
    if (_sharedConstants) {
        auto* dialect = _ctx->getLoadedDialect<IE::IEDialect>();
        VPUX_THROW_UNLESS(dialect != nullptr, "Got NULL pointer for IEDialect");

        const auto rawBuffer = StringRef(origNode->get_data_ptr<char>(), numElems * elemTypeSize.count());
        value = mlir::OpaqueElementsAttr::get(dialect, tensorType, rawBuffer);
    } else {
        const auto rawBuffer = makeArrayRef(origNode->get_data_ptr<char>(), numElems * elemTypeSize.count());

        bool isSplatBuffer = false;
        VPUX_THROW_UNLESS(mlir::DenseElementsAttr::isValidRawBuffer(tensorType, rawBuffer, isSplatBuffer),
                          "Constant node '{0}' has invalid buffer", origNode->get_friendly_name());

        value = mlir::DenseElementsAttr::getFromRawBuffer(tensorType, rawBuffer, isSplatBuffer);
    }

    auto op = builder.create<IE::ConstantOp>(createLocation(origNode), tensorType, value);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Convert>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Convert node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto dstType = importElemType(origNode->get_destination_type());
    const auto dstTypeAttr = mlir::TypeAttr::get(dstType);

    auto op = builder.create<IE::ConvertOp>(createLocation(origNode), inputs[0], dstTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Softmax>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Softmax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(axis));

    auto op = builder.create<IE::SoftMaxOp>(createLocation(origNode), inputs[0], axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Tile>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Tile node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TileOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Relu>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ReLUOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Split>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Split node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto num_splits = origNode->get_num_splits();
    const auto numSplitsAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(num_splits));

    auto op = builder.create<IE::SplitOp>(createLocation(origNode), inputs[0], inputs[1], numSplitsAttr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Power>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Power node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::PowerOp>(createLocation(origNode), inputs[0], inputs[1],
                                          importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Multiply>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Multiply node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MultiplyOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::MatMul>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::MatMulOp>(createLocation(origNode), inputs[0], inputs[1], origNode->get_transpose_a(),
                                           origNode->get_transpose_b());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Convolution>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrStride = getInt32ArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getInt32ArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getInt32ArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getInt32ArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::ConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, attrStride,
                                                attrPadsBegin, attrPadsEnd, attrDilation);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::GroupConvolution>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrStride = getInt32ArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getInt32ArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getInt32ArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getInt32ArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::GroupConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr,
                                                     attrStride, attrPadsBegin, attrPadsEnd, attrDilation, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::ConvolutionBackpropData>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 2) || (inputs.size() == 3),
                      "nGraph node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto attrStride = getInt32ArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getInt32ArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getInt32ArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getInt32ArrayAttr(_ctx, origNode->get_dilations());
    const auto attrOutputPadding = getInt32ArrayAttr(_ctx, origNode->get_output_padding());

    if (inputs.size() == 2) {
        auto op =
                builder.create<IE::DeconvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, attrStride,
                                                    attrPadsBegin, attrPadsEnd, attrDilation, attrOutputPadding);
        addOutputs(origNode, op);
    } else if (inputs.size() == 3) {
        auto op = builder.create<IE::DeconvolutionOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                      attrStride, attrPadsBegin, attrPadsEnd, attrDilation,
                                                      attrOutputPadding);
        addOutputs(origNode, op);
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::AvgPool>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getInt32ArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getInt32ArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getInt32ArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getInt32ArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());

    auto op = builder.create<IE::AvgPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::MaxPool>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getInt32ArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getInt32ArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getInt32ArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getInt32ArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());

    auto op = builder.create<IE::MaxPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Add>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::AddOp>(createLocation(origNode), inputs[0], inputs[1],
                                        importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Divide>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::DivideOp>(createLocation(origNode), inputs[0], inputs[1],
                                           importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::SquaredDifference>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SquaredDifferenceOp>(createLocation(origNode), inputs[0], inputs[1],
                                                      importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::FloorMod>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::FloorModOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::PriorBox>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& attrs = origNode->get_attrs();

    auto op = builder.create<IE::PriorBoxOp>(
            createLocation(origNode), inputs[0], inputs[1], getFP32ArrayAttr(_ctx, attrs.min_size),
            getFP32ArrayAttr(_ctx, attrs.max_size), getFP32ArrayAttr(_ctx, attrs.aspect_ratio),
            mlir::BoolAttr::get(_ctx, attrs.flip), mlir::BoolAttr::get(_ctx, attrs.clip), getFP32Attr(_ctx, attrs.step),
            getFP32Attr(_ctx, attrs.offset), getFP32ArrayAttr(_ctx, attrs.variance),
            mlir::BoolAttr::get(_ctx, attrs.scale_all_sizes), getFP32ArrayAttr(_ctx, attrs.fixed_ratio),
            getFP32ArrayAttr(_ctx, attrs.fixed_size), getFP32ArrayAttr(_ctx, attrs.density));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::PriorBoxClustered>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& attrs = origNode->get_attrs();

    auto op = builder.create<IE::PriorBoxClusteredOp>(
            createLocation(origNode), inputs[0], inputs[1], getFP32ArrayAttr(_ctx, attrs.widths),
            getFP32ArrayAttr(_ctx, attrs.heights), mlir::BoolAttr::get(_ctx, attrs.clip),
            getFP32Attr(_ctx, attrs.step_widths), getFP32Attr(_ctx, attrs.step_heights),
            getFP32Attr(_ctx, attrs.offset), getFP32ArrayAttr(_ctx, attrs.variances));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Gather>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Gather node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::GatherOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Reshape>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Reshape node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op =
            builder.create<IE::ReshapeOp>(createLocation(origNode), inputs[0], inputs[1], origNode->get_special_zero());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Minimum>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Minimum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MinimumOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Maximum>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Maximum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MaximumOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Clamp>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Clamp node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto min = origNode->get_min();
    const auto max = origNode->get_max();
    const auto minAttr = getFP32Attr(_ctx, checked_cast<float>(min));
    const auto maxAttr = getFP32Attr(_ctx, checked_cast<float>(max));

    auto op = builder.create<IE::ClampOp>(createLocation(origNode), inputs[0], minAttr, maxAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Proposal>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Proposal node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& proposalParam = origNode->get_attrs();
    const auto proposalParamAttr = importProposalAttrs(proposalParam);

    auto op = builder.create<IE::ProposalOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                             proposalParamAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Unsqueeze>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::UnsqueezeOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::LRN>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto beta = origNode->get_beta();
    const auto bias = origNode->get_bias();
    const auto size = origNode->get_nsize();

    const auto alphaAttr = getFP64Attr(_ctx, checked_cast<double>(alpha));
    const auto betaAttr = getFP64Attr(_ctx, checked_cast<double>(beta));
    const auto biasAttr = getFP64Attr(_ctx, checked_cast<double>(bias));
    const auto sizeAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(size));

    auto op = builder.create<IE::LRNOp>(createLocation(origNode), inputs[0], inputs[1], alphaAttr, betaAttr, biasAttr,
                                        sizeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Sigmoid>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SigmoidOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Squeeze>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2, "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SqueezeOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Transpose>& origNode) {
    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Transpose node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TransposeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Tanh>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Tanh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TanhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Elu>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Elu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto alphaAttr = getFP32Attr(_ctx, checked_cast<float>(alpha));

    auto op = builder.create<IE::EluOp>(createLocation(origNode), inputs[0], alphaAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::HSwish>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph HSwish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HSwishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::FakeQuantize>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 5, "nGraph FakeQuantize node '{0}' has unsupported number of inputs '{1}'.",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_auto_broadcast();

    const auto levelsAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(origNode->get_levels()));

    auto op = builder.create<IE::FakeQuantizeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 inputs[4], levelsAttr, importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Exp>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Exp node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ExpOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::StridedSlice>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph StridedSlice node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto attrBeginMask = getInt64ArrayAttr(_ctx, origNode->get_begin_mask());
    auto attrEndMask = getInt64ArrayAttr(_ctx, origNode->get_end_mask());
    auto attrNewAxisMask = getInt64ArrayAttr(_ctx, origNode->get_new_axis_mask());
    auto attrShrinkAxisMask = getInt64ArrayAttr(_ctx, origNode->get_shrink_axis_mask());
    auto attrEllipsisAxisMask = getInt64ArrayAttr(_ctx, origNode->get_ellipsis_mask());

    auto op = builder.create<IE::StridedSliceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 attrBeginMask, attrEndMask, attrNewAxisMask, attrShrinkAxisMask,
                                                 attrEllipsisAxisMask);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset2::ROIPooling>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ROIPooling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputSize = getInt32ArrayAttr(_ctx, origNode->get_output_size());
    const auto spatialScaleAttr = getFP32Attr(_ctx, origNode->get_spatial_scale());
    const auto method = importROIPoolingMethod(origNode->get_method());

    auto op = builder.create<IE::ROIPoolingOp>(createLocation(origNode), inputs[0], inputs[1], outputSize,
                                               spatialScaleAttr, method);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Concat>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() >= 1, "nGraph Concat node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getSInt32Attr(_ctx, checked_cast<int32_t>(axis));

    auto op = builder.create<IE::ConcatOp>(createLocation(origNode), inputs, axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Interpolate>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto interpolateAttr = importInterpolateAttrs(origNode->get_attrs());

    auto op = builder.create<IE::InterpolateOp>(createLocation(origNode), inputs[0], inputs[1], interpolateAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::TopK>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph TopK node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getInt64Attr(_ctx, checked_cast<int64_t>(origNode->get_axis()));
    const auto modeAttr = importTopKMode(origNode->get_mode());
    const auto sortTypeAttr = importTopKSortType(origNode->get_sort_type());
    const auto indexElementTypeAttr = mlir::TypeAttr::get(importElemType(origNode->get_index_element_type()));

    auto op = builder.create<IE::TopKOp>(createLocation(origNode), inputs[0], inputs[1], axisAttr, modeAttr,
                                         sortTypeAttr, indexElementTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::RegionYolo>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph RegionYolo node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto coordAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(origNode->get_num_coords()));
    const auto classesAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(origNode->get_num_classes()));
    const auto regionsAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(origNode->get_num_regions()));
    const auto doSoftmaxAttr = mlir::BoolAttr::get(_ctx, origNode->get_do_softmax());
    const auto maskAttr = getInt64ArrayAttr(_ctx, origNode->get_mask());
    const auto axisAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(origNode->get_axis()));
    const auto axisEndAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(origNode->get_end_axis()));
    const auto anchorsAttr = getFP32ArrayAttr(_ctx, origNode->get_anchors());

    auto op = builder.create<IE::RegionYoloOp>(createLocation(origNode), inputs[0], coordAttr, classesAttr, regionsAttr,
                                               doSoftmaxAttr, maskAttr, axisAttr, axisEndAttr, anchorsAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset2::ReorgYolo>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ReorgYolo node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto strides = origNode->get_strides();

    VPUX_THROW_UNLESS(strides.size() == 2, "nGraph ReorgYolo node '{0}' has unsupported number of strides '{1}'",
                      origNode->get_friendly_name(), strides.size());
    VPUX_THROW_UNLESS(strides.front() == strides.back(),
                      "nGraph ReorgYolo node '{0}' has different strides '{1}' != '{2}'", origNode->get_friendly_name(),
                      strides.front(), strides.back());

    const auto strideAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(strides.front()));

    auto op = builder.create<IE::ReorgYoloOp>(createLocation(origNode), inputs[0], strideAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::DetectionOutput>& origNode) {
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::NormalizeL2>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Normalize node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto epsAttr = getFP32Attr(_ctx, checked_cast<float>(origNode->get_eps()));
    const auto epsModeAttr = importEpsMode(origNode->get_eps_mode());

    auto op = builder.create<IE::NormalizeL2Op>(createLocation(origNode), inputs[0], inputs[1], epsAttr, epsModeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::PRelu>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph PRelu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::PReluOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::Swish>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Swish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SwishOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::GRN>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph GRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto biasAttr = getFP32Attr(_ctx, checked_cast<float>(origNode->get_bias()));

    auto op = builder.create<IE::GRNOp>(createLocation(origNode), inputs[0], biasAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Negative>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Negative node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::NegativeOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset1::CTCGreedyDecoder>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph CTCGreedyDecoder node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CTCGreedyDecoderOp>(createLocation(origNode), inputs[0], inputs[1],
                                                     origNode->get_ctc_merge_repeated());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ngraph::opset6::CTCGreedyDecoderSeqLen>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3,
                      "nGraph CTCGreedyDecoderSeqLen node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CTCGreedyDecoderSeqLenOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                           origNode->get_merge_repeated());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset6::Pad>& origNode) {
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
    const auto nodeName = mlir::Identifier::get(node->get_friendly_name(), _ctx);
    return mlir::NameLoc::get(nodeName);
}

//
// nGraph attributes importers
//

SmallVector<int64_t> NGraphImporter::importShape(const ngraph::PartialShape& shape) {
    VPUX_THROW_UNLESS(shape.rank().is_static(), "Dynamically ranked tensors are not supported");

    SmallVector<int64_t> out(checked_cast<size_t>(shape.rank().get_length()));
    for (const auto ind : irange(out.size())) {
        const auto& dim = shape[ind];
        out[ind] = dim.is_static() ? dim.get_length() : mlir::ShapedType::kDynamicSize;
    }

    return out;
}

mlir::Type NGraphImporter::importElemType(const ngraph::element::Type& elemType) {
    if (elemType == ngraph::element::f64) {
        return mlir::Float64Type::get(_ctx);
    } else if (elemType == ngraph::element::f32) {
        return mlir::Float32Type::get(_ctx);
    } else if (elemType == ngraph::element::f16) {
        return mlir::Float16Type::get(_ctx);
    } else if (elemType == ngraph::element::bf16) {
        return mlir::BFloat16Type::get(_ctx);
    } else if (elemType == ngraph::element::i64) {
        return getSInt64Type(_ctx);
    } else if (elemType == ngraph::element::u64) {
        return getUInt64Type(_ctx);
    } else if (elemType == ngraph::element::i32) {
        return getSInt32Type(_ctx);
    } else if (elemType == ngraph::element::u32) {
        return getUInt32Type(_ctx);
    } else if (elemType == ngraph::element::i16) {
        return getSInt16Type(_ctx);
    } else if (elemType == ngraph::element::u16) {
        return getUInt16Type(_ctx);
    } else if (elemType == ngraph::element::i8) {
        return getSInt8Type(_ctx);
    } else if (elemType == ngraph::element::u8) {
        return getUInt8Type(_ctx);
    } else {
        VPUX_THROW("Unsupported element type : {0}", elemType);
    }
}

mlir::RankedTensorType NGraphImporter::importTensor(const ngraph::PartialShape& shape,
                                                    const ngraph::element::Type& elemType) {
    return mlir::RankedTensorType::get(makeArrayRef(importShape(shape)), importElemType(elemType));
}

IE::AutoBroadcastTypeAttr NGraphImporter::importBroadcastType(ngraph::op::AutoBroadcastType bType) {
    switch (bType) {
    case ngraph::op::AutoBroadcastType::NONE:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    case ngraph::op::AutoBroadcastType::NUMPY:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::NUMPY);
    case ngraph::op::AutoBroadcastType::PDPD:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::PDPD);
    default:
        VPUX_THROW("Unknown AutoBroadcastType");
    }
}

IE::RoundingTypeAttr NGraphImporter::importRoundingType(ngraph::op::RoundingType roundingType) {
    switch (roundingType) {
    case ngraph::op::RoundingType::FLOOR:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::FLOOR);
    case ngraph::op::RoundingType::CEIL:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::CEIL);
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}

IE::EpsModeAttr NGraphImporter::importEpsMode(ngraph::op::EpsMode val) {
    switch (val) {
    case ngraph::op::EpsMode::ADD:
        return IE::EpsModeAttr::get(_ctx, IE::EpsMode::ADD);
    case ngraph::op::EpsMode::MAX:
        return IE::EpsModeAttr::get(_ctx, IE::EpsMode::MAX);
    default:
        VPUX_THROW("Unknown EpsMode");
    }
}

IE::TopKModeAttr NGraphImporter::importTopKMode(ngraph::op::TopKMode val) {
    switch (val) {
    case ngraph::op::TopKMode::MAX:
        return IE::TopKModeAttr::get(_ctx, IE::TopKMode::MAX);
    case ngraph::op::TopKMode::MIN:
        return IE::TopKModeAttr::get(_ctx, IE::TopKMode::MIN);
    default:
        VPUX_THROW("Unknown TopKMode");
    }
}

IE::TopKSortTypeAttr NGraphImporter::importTopKSortType(ngraph::op::TopKSortType val) {
    switch (val) {
    case ngraph::op::TopKSortType::NONE:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::NONE);
    case ngraph::op::TopKSortType::SORT_INDICES:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::SORT_INDICES);
    case ngraph::op::TopKSortType::SORT_VALUES:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::SORT_VALUES);
    default:
        VPUX_THROW("Unknown TopKSortType");
    }
}

IE::ProposalAttr NGraphImporter::importProposalAttrs(const ngraph::op::ProposalAttrs& val) {
    const auto baseSizeAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(val.base_size));
    const auto preNmsTopNAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(val.pre_nms_topn));
    const auto postNmsTopNAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(val.post_nms_topn));
    const auto nmsThreshNAttr = getFP32Attr(_ctx, checked_cast<float>(val.nms_thresh));
    const auto featStrideAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(val.feat_stride));
    const auto minSizeNAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(val.min_size));
    const auto ratioAttr = getFP32ArrayAttr(_ctx, val.ratio);
    const auto scaleAttr = getFP32ArrayAttr(_ctx, val.scale);
    const auto clipBeforeNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_before_nms);
    const auto clipAfterNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_after_nms);
    const auto normalizeAttr = mlir::BoolAttr::get(_ctx, val.normalize);
    const auto boxSizeScaleAttr = getFP32Attr(_ctx, checked_cast<float>(val.box_size_scale));
    const auto boxCoordinateScaleAttr = getFP32Attr(_ctx, checked_cast<float>(val.box_coordinate_scale));
    const auto frameworkAttr = mlir::StringAttr::get(_ctx, val.framework);
    const auto inferProbsAttr = mlir::BoolAttr::get(_ctx, val.infer_probs);

    return IE::ProposalAttr::get(baseSizeAttr, preNmsTopNAttr, postNmsTopNAttr, nmsThreshNAttr, featStrideAttr,
                                 minSizeNAttr, ratioAttr, scaleAttr, clipBeforeNmsAttr, clipAfterNmsAttr, normalizeAttr,
                                 boxSizeScaleAttr, boxCoordinateScaleAttr, frameworkAttr, inferProbsAttr, _ctx);
}

IE::InterpolateAttr NGraphImporter::importInterpolateAttrs(const ngraph::op::InterpolateAttrs& val) {
    const auto modeAttr = mlir::StringAttr::get(_ctx, val.mode).dyn_cast<IE::InterpolateModeAttr>();
    VPUX_THROW_UNLESS(modeAttr != nullptr, "Unsupported interpolate mode '{0}'", val.mode);

    const auto axesAttr = getInt64ArrayAttr(_ctx, val.axes);
    const auto alignCornersAttr = mlir::BoolAttr::get(_ctx, val.align_corners);
    const auto antialiasAttr = mlir::BoolAttr::get(_ctx, val.antialias);
    const auto padsBeginAttr = getInt32ArrayAttr(_ctx, val.pads_begin);
    const auto padsEndAttr = getInt32ArrayAttr(_ctx, val.pads_end);

    return IE::InterpolateAttr::get(axesAttr, modeAttr, alignCornersAttr, antialiasAttr, padsBeginAttr, padsEndAttr,
                                    _ctx);
}

IE::DetectionOutputAttr NGraphImporter::importDetectionOutputAttrs(const ngraph::op::DetectionOutputAttrs& val) {
    const auto numClassesAttr = getInt32Attr(_ctx, val.num_classes);
    const auto backgroundLabelIdAttr = getInt32Attr(_ctx, val.background_label_id);
    const auto topKAttr = getInt32Attr(_ctx, val.top_k);

    const auto varianceEncodedInTargetAttr = mlir::BoolAttr::get(_ctx, val.variance_encoded_in_target);

    const auto keepTopKAttr = getInt32ArrayAttr(_ctx, val.keep_top_k);
    const auto codeTypeAttr = mlir::StringAttr::get(_ctx, val.code_type);

    const auto shareLocationAttr = mlir::BoolAttr::get(_ctx, val.share_location);

    const auto nmsThresholdAttr = getFP32Attr(_ctx, val.nms_threshold);
    const auto confidenceThresholdAttr = getFP32Attr(_ctx, val.confidence_threshold);

    const auto clipAfterNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_after_nms);
    const auto clipBeforeNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_before_nms);
    const auto decreaseLabel_idAttr = mlir::BoolAttr::get(_ctx, val.decrease_label_id);
    const auto normalizedAttr = mlir::BoolAttr::get(_ctx, val.normalized);

    const auto inputHeightAttr = getUInt32Attr(_ctx, checked_cast<uint32_t>(val.input_height));
    const auto inputWidthAttr = getUInt32Attr(_ctx, checked_cast<uint32_t>(val.input_width));

    const auto objectnessScoreAttr = getFP32Attr(_ctx, val.objectness_score);

    return IE::DetectionOutputAttr::get(
            numClassesAttr, backgroundLabelIdAttr, topKAttr, varianceEncodedInTargetAttr, keepTopKAttr, codeTypeAttr,
            shareLocationAttr, nmsThresholdAttr, confidenceThresholdAttr, clipAfterNmsAttr, clipBeforeNmsAttr,
            decreaseLabel_idAttr, normalizedAttr, inputHeightAttr, inputWidthAttr, objectnessScoreAttr, _ctx);
}

IE::ROIPoolingMethodAttr NGraphImporter::importROIPoolingMethod(const std::string& method) {
    IE::ROIPoolingMethodAttr attr;
    if (method == "max") {
        attr = IE::ROIPoolingMethodAttr::get(_ctx, IE::ROIPoolingMethod::max);
    } else if (method == "bilinear") {
        attr = IE::ROIPoolingMethodAttr::get(_ctx, IE::ROIPoolingMethod::bilinear);
    } else {
        VPUX_THROW("Unknown ROIPoolingMethod");
    }
    return attr;
}

IE::PadModeAttr NGraphImporter::importPadMode(const ngraph::op::PadMode val) {
    IE::PadModeAttr attr;
    switch (val) {
    case ngraph::op::PadMode::CONSTANT:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::CONSTANT);
        break;
    case ngraph::op::PadMode::EDGE:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::EDGE);
        break;
    case ngraph::op::PadMode::REFLECT:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::REFLECT);
        break;
    case ngraph::op::PadMode::SYMMETRIC:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::SYMMETRIC);
        break;
    default:
        VPUX_THROW("Unknown PadMode");
    }
    return attr;
}

mlir::AffineMap importLayout(mlir::MLIRContext* ctx, InferenceEngine::Layout layout, size_t numDims) {
    if (numDims > 5) {
        return {};
    }

    switch (layout) {
    case InferenceEngine::Layout::ANY:
    case InferenceEngine::Layout::SCALAR:
    case InferenceEngine::Layout::C:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NCDHW:
        return {};
    case InferenceEngine::Layout::HWC:
        return DimsOrder::HWC.toAffineMap(ctx);
    case InferenceEngine::Layout::NHWC:
        return DimsOrder::NHWC.toAffineMap(ctx);
    case InferenceEngine::Layout::NDHWC:
        return DimsOrder::NDHWC.toAffineMap(ctx);

    default:
        VPUX_THROW("Unsupported layout '{0}'", layout);
    }
}

mlir::Type importPrecision(mlir::MLIRContext* ctx, const InferenceEngine::Precision& precision) {
    if (precision == InferenceEngine::Precision::FP32) {
        return mlir::Float32Type::get(ctx);
    } else if (precision == InferenceEngine::Precision::FP16) {
        return mlir::Float16Type::get(ctx);
    } else if (precision == InferenceEngine::Precision::I64) {
        return getSInt64Type(ctx);
    } else if (precision == InferenceEngine::Precision::U64) {
        return getUInt64Type(ctx);
    } else if (precision == InferenceEngine::Precision::I32) {
        return getSInt32Type(ctx);
    } else if (precision == InferenceEngine::Precision::U32) {
        return getUInt32Type(ctx);
    } else if (precision == InferenceEngine::Precision::I16) {
        return getSInt16Type(ctx);
    } else if (precision == InferenceEngine::Precision::U16) {
        return getUInt16Type(ctx);
    } else if (precision == InferenceEngine::Precision::I8) {
        return getSInt8Type(ctx);
    } else if (precision == InferenceEngine::Precision::U8) {
        return getUInt8Type(ctx);
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", precision);
    }
}

mlir::MemRefType importBuffer(mlir::MLIRContext* ctx, const InferenceEngine::TensorDesc& desc) {
    SmallVector<int64_t> shape(desc.getDims().size());
    std::copy(desc.getDims().begin(), desc.getDims().end(), shape.begin());

    const auto precision = importPrecision(ctx, desc.getPrecision());

    SmallVector<mlir::AffineMap> affineMaps;
    if (auto layout = importLayout(ctx, desc.getLayout(), desc.getDims().size())) {
        affineMaps.push_back(layout);
    }

    return mlir::MemRefType::get(shape, precision, affineMaps);
}

std::string getValidOutputName(const std::shared_ptr<ngraph::op::Result>& result) {
    const auto* resultInput = result->get_input_node_ptr(0);
    std::string portSuffix;
    if (resultInput->get_output_size() != 1) {
        portSuffix = "." + std::to_string(result->get_input_source_output(0).get_index());
    }
    return resultInput->get_friendly_name() + portSuffix;
}

//
// runNGraphPasses
//

void runNGraphPasses(std::shared_ptr<ngraph::Function> netGraph) {
    const auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    passConfig->disable<ngraph::pass::HSwishDecomposition>();
    passConfig->disable<ngraph::pass::HSigmoidDecomposition>();
    passConfig->disable<ngraph::pass::ConvertMinimum>();
    passConfig->disable<ngraph::pass::ConvertSubtract>();
    passConfig->disable<ngraph::pass::ConvertDivide>();
    passConfig->disable<ngraph::pass::ConvertNegative>();
    passConfig->disable<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();

    ngraph::pass::Manager manager(passConfig);
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::CommonOptimizations>();

    manager.run_passes(netGraph);
}

}  // namespace

//
// importNetwork
//

mlir::OwningModuleRef vpux::IE::importNetwork(mlir::MLIRContext* ctx, InferenceEngine::CNNNetwork cnnNet,
                                              bool sharedConstants, Logger log) {
    log.setName("IE::FrontEnd");

    log.trace("Load IE::FrontEnd dependent Dialects");
    ctx->loadDialect<IE::IEDialect>();

    const auto netGraph = cnnNet.getFunction();
    VPUX_THROW_UNLESS(netGraph != nullptr, "Old IR versions (prior v10) are not supported : {0}", cnnNet.getName());

    log.trace("Run nGraph passes");
    runNGraphPasses(netGraph);

    const auto inputsInfo = cnnNet.getInputsInfo();
    const auto outputsInfo = cnnNet.getOutputsInfo();

    const auto mainFuncName = mlir::FlatSymbolRefAttr::get(ctx, "main");

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef(cnnNet.getName()));

    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(ctx), mainFuncName);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), &builderLog);
    for (const auto& param : netGraph->get_parameters()) {
        const auto& inputName = param->get_friendly_name();
        const auto& userInput = inputsInfo.at(inputName);
        const auto& userDesc = userInput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(ctx, inputName);
        const auto userTypeAttr = mlir::TypeAttr::get(importBuffer(ctx, userDesc));

        inputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), &builderLog);
    for (const auto& result : netGraph->get_results()) {
        const auto& resultName = getValidOutputName(result);
        const auto& userOutput = outputsInfo.at(resultName);
        const auto& userDesc = userOutput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(ctx, resultName);
        const auto userTypeAttr = mlir::TypeAttr::get(importBuffer(ctx, userDesc));

        outputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }

    NGraphImporter importer(ctx, netGraph, sharedConstants, log);
    importer.buildMainFunc(builder, mainFuncName.getValue());

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    return module;
}
