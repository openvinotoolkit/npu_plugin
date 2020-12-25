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
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

using namespace vpux;

namespace {

class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, const std::shared_ptr<const ngraph::Function>& netGraph, Logger log)
            : _ctx(ctx), _netGraph(netGraph), _log(log) {
    }

public:
    mlir::FuncOp buildMainFunc(StringRef funcName);

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
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::AvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::MaxPool>& origNode);
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
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::FakeQuantize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::MatMul>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Tanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Exp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::HSwish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Transpose>& origNode);

    template <class NodeType>
    void parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
        parseNode(builder, std::dynamic_pointer_cast<NodeType>(origNode));
    }

    void parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
    }

private:
    SmallVector<mlir::Value, 4> getInputs(const OrigNodePtr& node);
    void addOutputs(const OrigNodePtr& node, mlir::Operation* op);
    mlir::Location createLocation(const OrigNodePtr& node);

private:
    static SmallVector<int64_t, 4> importShape(const ngraph::PartialShape& shape);
    mlir::Type importElemType(const ngraph::element::Type& elemType);
    mlir::RankedTensorType importTensor(const ngraph::PartialShape& shape, const ngraph::element::Type& elemType);
    IE::AutoBroadcastTypeAttr importBroadcastType(ngraph::op::AutoBroadcastType bType);
    IE::RoundingTypeAttr importRoundingType(ngraph::op::RoundingType roundingType);

private:
    mlir::MLIRContext* _ctx = nullptr;
    std::shared_ptr<const ngraph::Function> _netGraph;
    Logger _log;

    NodeOutputMap _importedVals;
};

//
// buildMainFunc
//

mlir::FuncOp NGraphImporter::buildMainFunc(StringRef funcName) {
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
            MAP_ENTRY(ngraph::opset1::AvgPool),
            MAP_ENTRY(ngraph::opset1::MaxPool),
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
            MAP_ENTRY(ngraph::opset1::FakeQuantize),
            MAP_ENTRY(ngraph::opset1::MatMul),
            MAP_ENTRY(ngraph::opset1::Tanh),
            MAP_ENTRY(ngraph::opset1::Exp),
            MAP_ENTRY(ngraph::opset4::HSwish),
            MAP_ENTRY(ngraph::opset1::Transpose),
    };

#undef MAP_ENTRY

    SmallVector<mlir::Type, 1> inputTypes;
    inputTypes.reserve(_netGraph->get_parameters().size());
    for (const auto& param : _netGraph->get_parameters()) {
        inputTypes.push_back(importTensor(param->get_partial_shape(), param->get_element_type()));
    }

    SmallVector<mlir::Type, 1> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }

    const auto funcType = mlir::FunctionType::get(_ctx, makeArrayRef(inputTypes), makeArrayRef(outputTypes));

    auto func = mlir::FuncOp::create(mlir::UnknownLoc::get(_ctx), funcName, funcType);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    for (const auto& p : _netGraph->get_parameters() | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = p.index();

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

    SmallVector<mlir::Value, 4> funcOutputs;
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

    auto tensorType = importTensor(origNode->get_output_partial_shape(0), origNode->get_output_element_type(0));

    auto elemType = tensorType.getElementType();
    auto totalNumElems = tensorType.getNumElements();

    mlir::DenseElementsAttr value;
    if (elemType.isF32()) {
        const auto* ptr = origNode->get_data_ptr<ngraph::element::Type_t::f32>();
        value = mlir::DenseElementsAttr::get(tensorType, makeArrayRef(ptr, totalNumElems));
    } else if (elemType.isF16()) {
        const auto* ptr = origNode->get_data_ptr<ngraph::element::Type_t::f16>();
        value = mlir::DenseElementsAttr::get(tensorType, makeArrayRef(ptr, totalNumElems));
    } else if (elemType.isSignedInteger(64)) {
        const auto* ptr = origNode->get_data_ptr<ngraph::element::Type_t::i64>();
        value = mlir::DenseElementsAttr::get(tensorType, makeArrayRef(ptr, totalNumElems));
    } else if (elemType.isUnsignedInteger(64)) {
        const auto* ptr = origNode->get_data_ptr<ngraph::element::Type_t::u64>();
        value = mlir::DenseElementsAttr::get(tensorType, makeArrayRef(ptr, totalNumElems));
    } else {
        VPUX_THROW("Element type '{0}' is not supported for Constant operation", elemType);
    }

    auto* dialect = _ctx->getLoadedDialect<IE::IEDialect>();
    VPUX_THROW_UNLESS(dialect != nullptr, "Got NULL pointer for IEDialect");

    auto* op = dialect->materializeConstant(builder, value, tensorType, createLocation(origNode));
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

    auto op = builder.create<IE::SplitOp>(createLocation(origNode), inputs[0], inputs[1], numSplitsAttr);
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
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Add node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::AddOp>(createLocation(origNode), inputs[0], inputs[1],
                                        importBroadcastType(autob.m_type));
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
    const auto minAttr = getFP64Attr(_ctx, checked_cast<double>(min));
    const auto maxAttr = getFP64Attr(_ctx, checked_cast<double>(max));

    auto op = builder.create<IE::ClampOp>(createLocation(origNode), inputs[0], minAttr, maxAttr);

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
    VPUX_THROW_UNLESS(inputs.size() <= 2, "nGraph Transpose node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TransposeOp>(createLocation(origNode), inputs[0], inputs[1]);
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
    const auto alphaAttr = getFP64Attr(_ctx, checked_cast<double>(alpha));

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

//
// IR builder helpers
//

SmallVector<mlir::Value, 4> NGraphImporter::getInputs(const OrigNodePtr& node) {
    SmallVector<mlir::Value, 4> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        out.push_back(_importedVals.at(input.get_source_output()));
    }

    return out;
}

void NGraphImporter::addOutputs(const OrigNodePtr& node, mlir::Operation* op) {
    const auto results = op->getOpResults();

    VPUX_THROW_UNLESS(results.size() == node->get_output_size(),
                      "Mismatch between orignal Node '{0}' number of outputs '{1}' and created number of outputs '{2}'",
                      node->get_friendly_name(), node->get_output_size(), results.size());

    for (const auto& res : results) {
        _importedVals.emplace(node->output(res.getResultNumber()), res);
    }
}

mlir::Location NGraphImporter::createLocation(const OrigNodePtr& node) {
    const auto nodeName = mlir::Identifier::get(node->get_friendly_name(), _ctx);
    return mlir::NameLoc::get(nodeName, _ctx);
}

//
// nGraph attributes importers
//

SmallVector<int64_t, 4> NGraphImporter::importShape(const ngraph::PartialShape& shape) {
    VPUX_THROW_UNLESS(shape.rank().is_static(), "Dynamically ranked tensors are not supported");

    SmallVector<int64_t, 4> out(checked_cast<size_t>(shape.rank().get_length()));
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

mlir::AffineMap importLayout(mlir::MLIRContext* ctx, InferenceEngine::Layout layout) {
    switch (layout) {
    case InferenceEngine::Layout::ANY:
    case InferenceEngine::Layout::SCALAR:
    case InferenceEngine::Layout::C:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NCDHW:
        return {};
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
    SmallVector<int64_t, MAX_NUM_DIMS> shape(desc.getDims().size());
    std::copy(desc.getDims().begin(), desc.getDims().end(), shape.begin());

    const auto precision = importPrecision(ctx, desc.getPrecision());

    SmallVector<mlir::AffineMap, 1> affineMaps;
    if (auto layout = importLayout(ctx, desc.getLayout())) {
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

}  // namespace

mlir::OwningModuleRef vpux::IE::importNetwork(mlir::MLIRContext* ctx, InferenceEngine::CNNNetwork cnnNet, Logger log) {
    log.setName("IE::FrontEnd");

    log.trace("Load IE::FrontEnd dependent Dialects");
    ctx->loadDialect<IE::IEDialect>();

    const auto netGraph = cnnNet.getFunction();
    VPUX_THROW_UNLESS(netGraph != nullptr, "Old IR versions (prior v10) are not supported : {0}", cnnNet.getName());

    const auto inputsInfo = cnnNet.getInputsInfo();
    const auto outputsInfo = cnnNet.getOutputsInfo();

    const auto mainFuncName = mlir::FlatSymbolRefAttr::get("main", ctx);

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef(cnnNet.getName()));

    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(ctx), mainFuncName);
    IE::CNNNetworkOp::ensureTerminator(cnnOp.inputsInfo(), builder, cnnOp.getLoc());
    IE::CNNNetworkOp::ensureTerminator(cnnOp.outputsInfo(), builder, cnnOp.getLoc());

    builder.setInsertionPointToStart(&cnnOp.inputsInfo().front());
    for (const auto& param : netGraph->get_parameters()) {
        const auto& inputName = param->get_friendly_name();
        const auto& userInput = inputsInfo.at(inputName);
        const auto& userDesc = userInput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(inputName, ctx);
        const auto userTypeAttr = mlir::TypeAttr::get(importBuffer(ctx, userDesc));

        builder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }

    builder.setInsertionPointToStart(&cnnOp.outputsInfo().front());
    for (const auto& result : netGraph->get_results()) {
        const auto& resultName = getValidOutputName(result);
        const auto& userOutput = outputsInfo.at(resultName);
        const auto& userDesc = userOutput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(resultName, ctx);
        const auto userTypeAttr = mlir::TypeAttr::get(importBuffer(ctx, userDesc));

        builder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }

    NGraphImporter importer(ctx, netGraph, log);
    const auto mainFunc = importer.buildMainFunc(mainFuncName.getValue());
    module.push_back(mainFunc);

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    return module;
}
