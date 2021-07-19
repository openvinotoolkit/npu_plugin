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

#include "vpux/compiler/frontend/IE.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
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
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/type/element_type.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_minimum_to_power_and_max.hpp>
#include <transformations/op_conversions/convert_negative.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>
#include "legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp"

using namespace vpux;

namespace {

namespace opset_latest = ngraph::opset7;

class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ngraph::Function> netGraph, bool sharedConstants,
                   Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

public:
    mlir::FuncOp buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName, mlir::TimingScope& rootTiming);

private:
    using OrigNode = ngraph::Node;
    using OrigNodePtr = std::shared_ptr<OrigNode>;
    using NodeOutputMap = std::unordered_map<ngraph::Output<OrigNode>, mlir::Value>;

private:
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Constant>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convert>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Softmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tile>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Relu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Split>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Power>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Multiply>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GroupConvolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::AvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Gather>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Clamp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Elu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Reshape>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Squeeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Unsqueeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Minimum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Maximum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Add>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Divide>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SquaredDifference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FloorMod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Proposal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FakeQuantize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MatMul>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Exp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HSwish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Transpose>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::Interpolate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::RegionYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReorgYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::DetectionOutput>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::NormalizeL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Concat>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::StridedSlice>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::PRelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Swish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Negative>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::CTCGreedyDecoder>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::CTCGreedyDecoderSeqLen>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Pad>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LSTMCell>& origNode);

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
    IE::InterpolateAttr importInterpolateAttrs(const ngraph::op::v4::Interpolate::InterpolateAttrs& val);
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

mlir::FuncOp NGraphImporter::buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName,
                                           mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Import nGraph function");

    using Callback = void (NGraphImporter::*)(mlir::OpBuilder & builder, const OrigNodePtr& origNode);
    using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

#define MAP_ENTRY(_NodeType_) \
    { _NodeType_::type_info, &NGraphImporter::parseDispatch<_NodeType_> }

    static const DispatchMap dispatchMap{
            {ngraph::op::Parameter::type_info, &NGraphImporter::parseEmpty},
            {ngraph::op::Result::type_info, &NGraphImporter::parseEmpty},

            MAP_ENTRY(opset_latest::Constant),
            MAP_ENTRY(opset_latest::Convert),
            MAP_ENTRY(opset_latest::Softmax),
            MAP_ENTRY(opset_latest::Tile),
            MAP_ENTRY(opset_latest::Split),
            MAP_ENTRY(opset_latest::Power),
            MAP_ENTRY(opset_latest::Multiply),
            MAP_ENTRY(opset_latest::Relu),
            MAP_ENTRY(opset_latest::Convolution),
            MAP_ENTRY(opset_latest::GroupConvolution),
            MAP_ENTRY(opset_latest::ConvolutionBackpropData),
            MAP_ENTRY(opset_latest::AvgPool),
            MAP_ENTRY(opset_latest::MaxPool),
            MAP_ENTRY(ngraph::opset1::Gather),
            MAP_ENTRY(opset_latest::Clamp),
            MAP_ENTRY(opset_latest::Elu),
            MAP_ENTRY(opset_latest::Reshape),
            MAP_ENTRY(opset_latest::Squeeze),
            MAP_ENTRY(opset_latest::Sigmoid),
            MAP_ENTRY(opset_latest::LRN),
            MAP_ENTRY(opset_latest::Unsqueeze),
            MAP_ENTRY(opset_latest::Minimum),
            MAP_ENTRY(opset_latest::Maximum),
            MAP_ENTRY(opset_latest::Add),
            MAP_ENTRY(opset_latest::Divide),
            MAP_ENTRY(opset_latest::SquaredDifference),
            MAP_ENTRY(opset_latest::FloorMod),
            MAP_ENTRY(ngraph::opset1::Proposal),
            MAP_ENTRY(opset_latest::FakeQuantize),
            MAP_ENTRY(opset_latest::MatMul),
            MAP_ENTRY(opset_latest::Tanh),
            MAP_ENTRY(opset_latest::Exp),
            MAP_ENTRY(opset_latest::HSwish),
            MAP_ENTRY(opset_latest::Transpose),
            MAP_ENTRY(ngraph::opset4::Interpolate),
            MAP_ENTRY(ngraph::opset1::TopK),
            MAP_ENTRY(opset_latest::RegionYolo),
            MAP_ENTRY(opset_latest::ReorgYolo),
            MAP_ENTRY(opset_latest::DetectionOutput),
            MAP_ENTRY(opset_latest::NormalizeL2),
            MAP_ENTRY(opset_latest::Concat),
            MAP_ENTRY(opset_latest::ROIPooling),
            MAP_ENTRY(opset_latest::StridedSlice),
            MAP_ENTRY(opset_latest::PRelu),
            MAP_ENTRY(opset_latest::Swish),
            MAP_ENTRY(opset_latest::GRN),
            MAP_ENTRY(opset_latest::Negative),
            MAP_ENTRY(opset_latest::CTCGreedyDecoder),
            MAP_ENTRY(opset_latest::CTCGreedyDecoderSeqLen),
            MAP_ENTRY(opset_latest::Pad),
            MAP_ENTRY(opset_latest::LSTMCell),
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Constant>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Constant>::value,
                  "opset operation mismatch");
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

    auto op = builder.create<Const::DeclareOp>(createLocation(origNode), tensorType, Const::ContentAttr::get(value));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convert>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Convert>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Convert node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto dstType = importElemType(origNode->get_destination_type());
    const auto dstTypeAttr = mlir::TypeAttr::get(dstType);

    auto op = builder.create<IE::ConvertOp>(createLocation(origNode), inputs[0], dstTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Softmax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Softmax>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Softmax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(axis));

    auto op = builder.create<IE::SoftMaxOp>(createLocation(origNode), inputs[0], axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tile>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Tile>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Tile node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TileOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Relu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Relu>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ReLUOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Split>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Split>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Split node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto num_splits = origNode->get_num_splits();
    const auto numSplitsAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(num_splits));

    auto op = builder.create<IE::SplitOp>(createLocation(origNode), inputs[0], inputs[1], numSplitsAttr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Power>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Power>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Multiply>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Multiply node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MultiplyOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MatMul>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::MatMul>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::MatMulOp>(createLocation(origNode), inputs[0], inputs[1], origNode->get_transpose_a(),
                                           origNode->get_transpose_b());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convolution>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Convolution>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrStride = getInt32ArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getInt32ArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getInt32ArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getInt32ArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::ConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, attrStride,
                                                attrPadsBegin, attrPadsEnd, attrDilation, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::GroupConvolution>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::GroupConvolution>::value,
                  "opset operation mismatch");
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
                               const std::shared_ptr<opset_latest::ConvolutionBackpropData>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::ConvolutionBackpropData>::value,
                  "opset operation mismatch");
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::AvgPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::AvgPool>::value,
                  "opset operation mismatch");
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::MaxPool>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getInt32ArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getInt32ArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getInt32ArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getInt32ArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());

    auto op = builder.create<IE::MaxPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Add>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Add>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::AddOp>(createLocation(origNode), inputs[0], inputs[1],
                                        importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Divide>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Divide>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::SquaredDifference>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::FloorMod>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::FloorModOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Gather>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Gather>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Gather node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::GatherOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Reshape>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Reshape>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Reshape node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ReshapeOp>(createLocation(origNode), inputs[0], inputs[1],
                                            origNode->get_special_zero(), nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Minimum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Minimum>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Maximum>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Clamp>::value,
                  "opset operation mismatch");
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Proposal>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Unsqueeze>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::UnsqueezeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LRN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::LRN>::value,
                  "opset operation mismatch");
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Sigmoid>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SigmoidOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Squeeze>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Squeeze>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2, "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SqueezeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Transpose>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Transpose>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Transpose node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TransposeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tanh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Tanh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Tanh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TanhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Elu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Elu>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Elu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto alphaAttr = getFP32Attr(_ctx, checked_cast<float>(alpha));

    auto op = builder.create<IE::EluOp>(createLocation(origNode), inputs[0], alphaAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HSwish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::HSwish>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph HSwish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HSwishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FakeQuantize>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::FakeQuantize>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 5, "nGraph FakeQuantize node '{0}' has unsupported number of inputs '{1}'.",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_auto_broadcast();

    const auto levelsAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(origNode->get_levels()));

    auto op = builder.create<IE::FakeQuantizeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 inputs[4], levelsAttr, importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Exp>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Exp>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Exp node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ExpOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::StridedSlice>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::StridedSlice>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph StridedSlice node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto attrBeginMask = getInt64ArrayAttr(_ctx, origNode->get_begin_mask());
    auto attrEndMask = getInt64ArrayAttr(_ctx, origNode->get_end_mask());
    auto attrNewAxisMask = getInt64ArrayAttr(_ctx, origNode->get_new_axis_mask());
    auto attrShrinkAxisMask = getInt64ArrayAttr(_ctx, origNode->get_shrink_axis_mask());
    auto attrEllipsisAxisMask = getInt64ArrayAttr(_ctx, origNode->get_ellipsis_mask());

    auto op = builder.create<IE::StridedSliceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 nullptr, nullptr, nullptr, attrBeginMask, attrEndMask, attrNewAxisMask,
                                                 attrShrinkAxisMask, attrEllipsisAxisMask);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ROIPooling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::ROIPooling>::value,
                  "opset operation mismatch");
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Concat>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Concat>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() >= 1, "nGraph Concat node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getSInt32Attr(_ctx, checked_cast<int32_t>(axis));

    auto op = builder.create<IE::ConcatOp>(createLocation(origNode), inputs, axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::Interpolate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::Interpolate>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto interpolateAttr = importInterpolateAttrs(origNode->get_attrs());

    auto op = builder.create<IE::InterpolateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                nullptr, nullptr, nullptr, interpolateAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::TopK>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::TopK>::value,
                  "opset operation mismatch");
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::RegionYolo>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::RegionYolo>::value,
                  "opset operation mismatch");
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReorgYolo>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::ReorgYolo>::value,
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

    const auto strideAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(strides.front()));

    auto op = builder.create<IE::ReorgYoloOp>(createLocation(origNode), inputs[0], strideAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::DetectionOutput>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::DetectionOutput>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::NormalizeL2>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Normalize node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto epsAttr = getFP32Attr(_ctx, checked_cast<float>(origNode->get_eps()));
    const auto epsModeAttr = importEpsMode(origNode->get_eps_mode());

    auto op = builder.create<IE::NormalizeL2Op>(createLocation(origNode), inputs[0], inputs[1], epsAttr, epsModeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::PRelu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::PRelu>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph PRelu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::PReluOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Swish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::Swish>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Swish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SwishOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::GRN>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph GRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto biasAttr = getFP32Attr(_ctx, checked_cast<float>(origNode->get_bias()));

    auto op = builder.create<IE::GRNOp>(createLocation(origNode), inputs[0], biasAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Negative>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Negative>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Negative node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::NegativeOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::CTCGreedyDecoder>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::CTCGreedyDecoder>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v6::CTCGreedyDecoderSeqLen>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3,
                      "nGraph CTCGreedyDecoderSeqLen node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CTCGreedyDecoderSeqLenOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                           origNode->get_merge_repeated());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Pad>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Pad>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, opset_latest::LSTMCell>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 6, "nGraph LSTMCell node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph LSTMCell node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh", "tanh"}),
                      "nGraph LSTMCell node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    auto op = builder.create<IE::LSTMCellOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                             inputs[4], inputs[5], origNode->get_hidden_size());
    addOutputs(origNode, op);
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

IE::InterpolateAttr NGraphImporter::importInterpolateAttrs(const ngraph::op::v4::Interpolate::InterpolateAttrs& val) {
    // mode
    IE::InterpolateModeAttr modeAttr;
    switch (val.mode) {
    case ngraph::op::v4::Interpolate::InterpolateMode::nearest:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::nearest);
        break;
    case ngraph::op::v4::Interpolate::InterpolateMode::linear:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::linear);
        break;
    case ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::linear_onnx);
        break;
    case ngraph::op::v4::Interpolate::InterpolateMode::cubic:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::cubic);
        break;
    default:
        VPUX_THROW("Unsupported interpolate mode");
    }

    // shape calculation mode
    IE::InterpolateCalcModeAttr calcModeAttr;
    switch (val.shape_calculation_mode) {
    case ngraph::op::v4::Interpolate::ShapeCalcMode::sizes:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::sizes);
        break;
    case ngraph::op::v4::Interpolate::ShapeCalcMode::scales:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::scales);
        break;
    default:
        VPUX_THROW("Unsupported interpolate shape calculation mode");
    }

    // coordinate transformation mode
    IE::InterpolateCoordModeAttr coordModeAttr;
    switch (val.coordinate_transformation_mode) {
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::half_pixel);
        break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::pytorch_half_pixel);
        break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::asymmetric);
        break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::tf_half_pixel_for_nn);
        break;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::align_corners);
        break;
    default:
        VPUX_THROW("Unsupported interpolate coordinate transformation mode");
    }

    // coordinate transformation mode
    IE::InterpolateNearestModeAttr nearestModeAttr;
    switch (val.nearest_mode) {
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::round_prefer_floor);
        break;
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::round_prefer_ceil);
        break;
    case ngraph::op::v4::Interpolate::NearestMode::floor:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::floor);
        break;
    case ngraph::op::v4::Interpolate::NearestMode::ceil:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::ceil);
        break;
    case ngraph::op::v4::Interpolate::NearestMode::simple:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::simple);
        break;
    default:
        VPUX_THROW("Unsupported interpolate nearest mode");
    }

    const auto antialiasAttr = mlir::BoolAttr::get(_ctx, val.antialias);
    const auto padsBeginAttr = getInt32ArrayAttr(_ctx, val.pads_begin);
    const auto padsEndAttr = getInt32ArrayAttr(_ctx, val.pads_end);
    const auto cubeCoeffAttr = getFP32Attr(_ctx, (float)(val.cube_coeff));

    return IE::InterpolateAttr::get(modeAttr, calcModeAttr, coordModeAttr, nearestModeAttr, antialiasAttr,
                                    padsBeginAttr, padsEndAttr, cubeCoeffAttr, _ctx);
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

mlir::RankedTensorType importUserTensor(mlir::MLIRContext* ctx, const InferenceEngine::TensorDesc& desc) {
    const Shape shape(desc.getDims().begin(), desc.getDims().end());
    const auto precision = importPrecision(ctx, desc.getPrecision());
    return getTensorType(shape.raw(), precision, DimsOrder::fromIE(desc.getLayout()));
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

void runNGraphPasses(const std::shared_ptr<ngraph::Function>& netGraph, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Common nGraph passes");

    const auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    passConfig->disable<ngraph::pass::HSwishDecomposition>();
    passConfig->disable<ngraph::pass::HSigmoidDecomposition>();
    passConfig->disable<ngraph::pass::ConvertMinimum>();
    passConfig->disable<ngraph::pass::ConvertSubtract>();
    passConfig->disable<ngraph::pass::ConvertDivide>();
    passConfig->disable<ngraph::pass::ConvertNegative>();
    passConfig->disable<ngraph::pass::LSTMCellDecomposition>();
    passConfig->disable<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    passConfig->disable<ngraph::pass::ConvertStridedSliceToCropMatcher>();

    ngraph::pass::Manager manager(passConfig);
    manager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::CommonOptimizations>();

    manager.run_passes(netGraph);
}

//
// addCNNNetworkOp
//

void addCNNNetworkOp(mlir::OpBuilder& builder, mlir::FlatSymbolRefAttr mainFuncName, InferenceEngine::CNNNetwork cnnNet,
                     const std::shared_ptr<ngraph::Function>& netGraph, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Add CNNNetwork Operation");

    const auto inputsInfo = cnnNet.getInputsInfo();
    const auto outputsInfo = cnnNet.getOutputsInfo();

    auto* ctx = builder.getContext();

    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(ctx), mainFuncName);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), builder.getListener());
    for (const auto& param : netGraph->get_parameters()) {
        const auto& inputName = param->get_friendly_name();
        const auto& userInput = inputsInfo.at(inputName);
        const auto& userDesc = userInput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(ctx, inputName);
        const auto userTypeAttr = mlir::TypeAttr::get(importUserTensor(ctx, userDesc));

        inputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), builder.getListener());
    for (const auto& result : netGraph->get_results()) {
        const auto& resultName = getValidOutputName(result);
        const auto& userOutput = outputsInfo.at(resultName);
        const auto& userDesc = userOutput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(ctx, resultName);
        const auto userTypeAttr = mlir::TypeAttr::get(importUserTensor(ctx, userDesc));

        outputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }
}

}  // namespace

//
// importNetwork
//

mlir::OwningModuleRef vpux::IE::importNetwork(mlir::MLIRContext* ctx, InferenceEngine::CNNNetwork cnnNet,
                                              bool sharedConstants, mlir::TimingScope& rootTiming, Logger log) {
    log.setName("IE::FrontEnd");

    log.trace("Load IE::FrontEnd dependent Dialects");
    ctx->loadDialect<IE::IEDialect>();

    const auto netGraph = cnnNet.getFunction();
    VPUX_THROW_UNLESS(netGraph != nullptr, "Old IR versions (prior v10) are not supported : {0}", cnnNet.getName());

    log.trace("Run common nGraph passes");
    runNGraphPasses(netGraph, rootTiming);

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef(cnnNet.getName()));
    const auto mainFuncName = mlir::FlatSymbolRefAttr::get(ctx, "main");

    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    log.trace("Add CNNNetwork Operation");
    addCNNNetworkOp(builder, mainFuncName, cnnNet, netGraph, rootTiming);

    log.trace("Import nGraph function");
    NGraphImporter importer(ctx, netGraph, sharedConstants, log);
    importer.buildMainFunc(builder, mainFuncName.getValue(), rootTiming);

    log.trace("Validate MLIR module");
    auto finalTiming = rootTiming.nest("Validate MLIR module");
    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    return module;
}
