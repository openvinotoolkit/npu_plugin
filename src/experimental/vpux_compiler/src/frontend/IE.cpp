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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/scalars.hpp"

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
#include <ngraph/opsets/opset1.hpp>
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
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Softmax>& origNode);

    template <class NodeType>
    void parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
        parseNode(builder, std::dynamic_pointer_cast<NodeType>(origNode));
    }

    void parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
    }

private:
    SmallVector<mlir::Value, 4> getInputs(const OrigNodePtr& node);
    void addOutputs(const OrigNodePtr& node, mlir::ValueRange vals);

private:
    static SmallVector<int64_t, 4> importShape(const ngraph::PartialShape& shape);
    mlir::Type importElemType(const ngraph::element::Type& elemType);
    mlir::RankedTensorType importTensor(const ngraph::PartialShape& shape, const ngraph::element::Type& elemType);
    mlir::Location createLocation(const OrigNodePtr& node);

private:
    mlir::MLIRContext* _ctx = nullptr;
    std::shared_ptr<const ngraph::Function> _netGraph;
    Logger _log;

    NodeOutputMap _importedVals;

private:
    using Callback = void (NGraphImporter::*)(mlir::OpBuilder& builder, const OrigNodePtr& origNode);
    using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

    static const DispatchMap dispatchMap;
};

#define MAP_ENTRY(_NodeType_) \
    { _NodeType_::type_info, &NGraphImporter::parseDispatch<_NodeType_> }

const NGraphImporter::DispatchMap NGraphImporter::dispatchMap{
        {ngraph::op::Parameter::type_info, &NGraphImporter::parseEmpty},
        {ngraph::op::Result::type_info, &NGraphImporter::parseEmpty},

        MAP_ENTRY(ngraph::opset1::Constant),
        MAP_ENTRY(ngraph::opset1::Softmax),
};

#undef MAP_ENTRY

mlir::FuncOp NGraphImporter::buildMainFunc(StringRef funcName) {
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

    const auto funcType = mlir::FunctionType::get(makeArrayRef(inputTypes), makeArrayRef(outputTypes), _ctx);

    auto func = mlir::FuncOp::create(mlir::UnknownLoc::get(_ctx), funcName, funcType);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    for (const auto& p : _netGraph->get_parameters() | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = p.index();

        _log.trace("Convert network Parameter {0}", paramNode->get_friendly_name());

        const auto funcInputVal = func.getArgument(paramIndex);
        addOutputs(paramNode, {funcInputVal});
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
    addOutputs(origNode, op->getResults());
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset1::Softmax>& origNode) {
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Softmax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getInt32Attr(_ctx, checked_cast<uint32_t>(axis));

    auto op = builder.create<IE::SoftMaxOp>(createLocation(origNode), inputs[0], axisAttr);
    addOutputs(origNode, {op.getResult()});
}

SmallVector<mlir::Value, 4> NGraphImporter::getInputs(const OrigNodePtr& node) {
    SmallVector<mlir::Value, 4> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        out.push_back(_importedVals.at(input.get_source_output()));
    }

    return out;
}

void NGraphImporter::addOutputs(const OrigNodePtr& node, mlir::ValueRange vals) {
    VPUX_THROW_UNLESS(vals.size() == node->get_output_size(),
                      "Mismatch between orignal Node '{0}' number of outputs '{1}' and created number of outputs '{2}'",
                      node->get_friendly_name(), node->get_output_size(), vals.size());

    for (const auto& p : make_range(vals) | indexed) {
        _importedVals.emplace(node->output(p.index()), p.value());
    }
}

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

mlir::Location NGraphImporter::createLocation(const OrigNodePtr& node) {
    const auto nodeName = mlir::Identifier::get(node->get_friendly_name(), _ctx);
    return mlir::NameLoc::get(nodeName, _ctx);
}

IE::LayoutAttr importLayout(mlir::MLIRContext* ctx, InferenceEngine::Layout layout) {
#define CASE(_l_)                      \
    case InferenceEngine::Layout::_l_: \
        return IE::LayoutAttr::get(IE::Layout::_l_, ctx)

    switch (layout) {
        CASE(ANY);
        CASE(SCALAR);
        CASE(C);
        CASE(NC);
        CASE(CHW);
        CASE(NCHW);
        CASE(NHWC);
        CASE(NCDHW);
        CASE(NDHWC);
    default:
        VPUX_THROW("Unsupported layout {0}", layout);
    }

#undef CASE
}

mlir::TypeAttr importPrecision(mlir::MLIRContext* ctx, const InferenceEngine::Precision& precision) {
    if (precision == InferenceEngine::Precision::FP32) {
        return mlir::TypeAttr::get(mlir::Float32Type::get(ctx));
    } else if (precision == InferenceEngine::Precision::FP16) {
        return mlir::TypeAttr::get(mlir::Float16Type::get(ctx));
    } else if (precision == InferenceEngine::Precision::I64) {
        return mlir::TypeAttr::get(getSInt64Type(ctx));
    } else if (precision == InferenceEngine::Precision::U64) {
        return mlir::TypeAttr::get(getUInt64Type(ctx));
    } else if (precision == InferenceEngine::Precision::I32) {
        return mlir::TypeAttr::get(getSInt32Type(ctx));
    } else if (precision == InferenceEngine::Precision::I16) {
        return mlir::TypeAttr::get(getSInt16Type(ctx));
    } else if (precision == InferenceEngine::Precision::U16) {
        return mlir::TypeAttr::get(getUInt16Type(ctx));
    } else if (precision == InferenceEngine::Precision::I8) {
        return mlir::TypeAttr::get(getSInt8Type(ctx));
    } else if (precision == InferenceEngine::Precision::U8) {
        return mlir::TypeAttr::get(getUInt8Type(ctx));
    } else {
        VPUX_THROW("Unsupported precision : {0}", precision);
    }
}

}  // namespace

vpux::IE::FrontEnd::FrontEnd(mlir::MLIRContext* ctx, Logger log): _ctx(ctx), _log(log) {
    _log.setName("IE::FrontEnd");

    _log.trace("Load IE::FrontEnd dependent Dialects");
    _ctx->loadDialect<IE::IEDialect>();
}

mlir::OwningModuleRef vpux::IE::FrontEnd::importNetwork(InferenceEngine::CNNNetwork cnnNet) const {
    const auto netGraph = cnnNet.getFunction();
    VPUX_THROW_UNLESS(netGraph != nullptr, "Old IR versions (prior v10) are not supported : {0}", cnnNet.getName());

    const auto inputsInfo = cnnNet.getInputsInfo();
    const auto outputsInfo = cnnNet.getOutputsInfo();

    const auto mainFuncName = mlir::FlatSymbolRefAttr::get("main", _ctx);

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(_ctx));

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(_ctx),
                                                  mlir::StringAttr::get(cnnNet.getName(), _ctx), mainFuncName);

    cnnOp.inputsInfo().push_back(new mlir::Block);
    builder.setInsertionPointToStart(&cnnOp.inputsInfo().front());
    for (const auto& param : netGraph->get_parameters()) {
        const auto& inputName = param->get_friendly_name();
        const auto& userInput = inputsInfo.at(inputName);
        const auto& userDesc = userInput->getTensorDesc();

        builder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(_ctx), mlir::StringAttr::get(inputName, _ctx),
                                       importPrecision(_ctx, userDesc.getPrecision()),
                                       importLayout(_ctx, userDesc.getLayout()));
    }
    builder.create<IE::EndOp>(mlir::UnknownLoc::get(_ctx));

    cnnOp.outputsInfo().push_back(new mlir::Block);
    builder.setInsertionPointToStart(&cnnOp.outputsInfo().front());
    for (const auto& result : netGraph->get_results()) {
        const auto* resultInput = result->get_input_node_ptr(0);
        const auto& resultName = resultInput->get_friendly_name();
        const auto& userOutput = outputsInfo.at(resultName);
        const auto& userDesc = userOutput->getTensorDesc();

        builder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(_ctx), mlir::StringAttr::get(resultName, _ctx),
                                       importPrecision(_ctx, userDesc.getPrecision()),
                                       importLayout(_ctx, userDesc.getLayout()));
    }
    builder.create<IE::EndOp>(mlir::UnknownLoc::get(_ctx));

    NGraphImporter importer(_ctx, netGraph, _log);
    const auto mainFunc = importer.buildMainFunc(mainFuncName.getValue());
    module.push_back(mainFunc);

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    return module;
}
