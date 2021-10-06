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

#include "vpux/compiler/backend/IE.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/IE/hash.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>

#include <ie_common.h>
#include <ie_layouts.h>
#include <ie_precision.hpp>

#include <cpp/ie_cnn_network.h>

#include <precision_utils.h>

#include <unordered_map>

//#include <ngraph/opsets/opset7.hpp>
#include <ngraph/op/abs.hpp>
#include <ngraph/op/parameter.hpp>
#include "ngraph/function.hpp"
#include "ngraph/shape.hpp"
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/type/element_type.hpp>

using namespace vpux;

namespace {

    using OrigNode = mlir::Operation;
    using OrigNodePtr = OrigNode *;
    using NodeMap = std::unordered_map<mlir::Operation *, std::shared_ptr<ngraph::Node>>;
class NGraphExporter final {
public:
    NGraphExporter() = default;

public:
    ngraph::op::AutoBroadcastType exportBroadcastType(IE::AutoBroadcastType bType);
    InferenceEngine::Precision exportPrecision(mlir::MLIRContext* ctx, mlir::Type type);
    InferenceEngine::TensorDesc exportUserTensor(llvm::SmallVector<IE::DataInfoOp> inputsInfo);
    ngraph::element::Type toNGraphType(InferenceEngine::Precision precision);
    std::shared_ptr<ngraph::Function> exportToNgraph(IE::CNNNetworkOp netOp, mlir::FuncOp netFunc);
private:
    std::shared_ptr<ngraph::Function> _netGraph;
    std::shared_ptr<ngraph::Node> parseNode(IE::AddOp origNode, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::SoftMaxOp origNode, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::PowerOp origNode, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::NegativeOp origNode, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(mlir::ReturnOp origNode, ngraph::OutputVector &inputs);
    template <class NodeType>
    std::shared_ptr<ngraph::Node> parseDispatch(OrigNodePtr origNode, ngraph::OutputVector &inputs) {
        return parseNode(llvm::dyn_cast<NodeType>(*origNode), inputs);
    }

    void parseEmpty(OrigNodePtr) {
    }
public:
    int _inputIndex = 0;
    NodeMap _importedVals;
    ngraph::ParameterVector _params;
};

std::shared_ptr<ngraph::Function> NGraphExporter::exportToNgraph(IE::CNNNetworkOp netOp, mlir::FuncOp netFunc)
{
    using Callback = std::shared_ptr<ngraph::Node> (NGraphExporter::*)(OrigNodePtr origNode, ngraph::OutputVector &inputs);
    using DispatchMap = std::map<std::string, Callback>;

#define MAP_ENTRY(_OpName_, _OpType_) \
    { _OpName_, &NGraphExporter::parseDispatch<_OpType_> }

    static DispatchMap dispatchMap {
            MAP_ENTRY("IE.Add", IE::AddOp),
            MAP_ENTRY("IE.SoftMax", IE::SoftMaxOp),
            MAP_ENTRY("IE.Power", IE::PowerOp),
            MAP_ENTRY("IE.Negative", IE::NegativeOp),
            MAP_ENTRY("std.return", mlir::ReturnOp),
    };
#undef MAP_ENTRY

    mlir::Block &block = *(netFunc.body().getBlocks().begin());
    block.walk([&](mlir::Operation *op) {
        std::shared_ptr<ngraph::Node> ngNode;
        ngraph::OutputVector inputs;
        for (unsigned i = 0; i < op->getNumOperands(); i++)
        {
            mlir::Operation *sourceOp = op->getOperand(i).getDefiningOp();
            if (sourceOp == nullptr)
            {
                llvm::SmallVector<IE::DataInfoOp> inputsInfo = to_small_vector(netOp.inputsInfo().getOps<IE::DataInfoOp>());
                InferenceEngine::TensorDesc tensor = exportUserTensor(inputsInfo);
                ngraph::Shape ngShape{tensor.getDims().begin(), tensor.getDims().end()};
                std::shared_ptr<ngraph::opset7::Parameter> par = 
                    std::make_shared<ngraph::opset7::Parameter>(toNGraphType(tensor.getPrecision()), ngShape);
                auto nameAttr = inputsInfo[_inputIndex]->getAttr("name");
                auto nameVal = nameAttr.dyn_cast<mlir::StringAttr>().getValue();
                par->set_friendly_name(nameVal.str());
                _inputIndex++;
                ngNode = par;
                _importedVals.insert({nullptr, ngNode});
                inputs.push_back(ngraph::Output<ngraph::Node>(ngNode));
                _params.push_back(par);
            }
            else
                inputs.push_back(ngraph::Output<ngraph::Node>(_importedVals.at(sourceOp)));
        }

        const auto dispatchIt = dispatchMap.find(op->getName().getStringRef().str());

        const auto parser = dispatchIt->second;
        ngNode = (this->*parser)(op, inputs);
        _importedVals.insert({op, ngNode});

    });
    return _netGraph;
}

//
// Parsers
//
std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::AddOp op, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(op.auto_broadcast());
    return std::make_shared<ngraph::opset7::Add>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}
std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::SoftMaxOp op, ngraph::OutputVector &inputs)
{
    auto axisIndVal = op.axisInd();
    return std::make_shared<ngraph::opset7::Softmax>(inputs.at(0), axisIndVal);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::PowerOp op, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(op.auto_broadcast());
    return std::make_shared<ngraph::opset7::Power>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::NegativeOp, ngraph::OutputVector &inputs)
{
    //op = op;
    return std::make_shared<ngraph::opset7::Negative>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(mlir::ReturnOp, ngraph::OutputVector &inputs)
{
    //op = op;
    std::shared_ptr<ngraph::Node> ngNode = std::make_shared<ngraph::opset7::Result>(inputs.at(0));
    _netGraph = std::make_shared<ngraph::Function>(ngraph::OutputVector{ngNode}, _params);
    return ngNode;
}

InferenceEngine::TensorDesc NGraphExporter::exportUserTensor(llvm::SmallVector<IE::DataInfoOp> inputsInfo) {
    auto userTypeAttr = inputsInfo[_inputIndex]->getAttr("userType");
    auto userTypeAttr2 = userTypeAttr.dyn_cast<mlir::TypeAttr>();
    auto userType = userTypeAttr2.getValue();
    const mlir::RankedTensorType& rankedTensorType = userType.dyn_cast<mlir::RankedTensorType>();
    const Shape shape = rankedTensorType.getShape();
    InferenceEngine::SizeVector dims;
    for (auto ddim : shape)
        dims.push_back(ddim);
    const mlir::Type elementType = rankedTensorType.getElementType();
    const InferenceEngine::Precision precision = exportPrecision(elementType.getContext(), elementType);
    DimsOrder dimsOrder = DimsOrder::fromType(rankedTensorType);
    InferenceEngine::Layout layout = dimsOrder.toIE();
    return InferenceEngine::TensorDesc{precision, dims, layout};
}

ngraph::op::AutoBroadcastType NGraphExporter::exportBroadcastType(IE::AutoBroadcastType bType) {
    switch (bType) {
    case IE::AutoBroadcastType::NONE_OR_EXPLICIT:
        return ngraph::op::AutoBroadcastType::NONE;
    case IE::AutoBroadcastType::NUMPY:
        return ngraph::op::AutoBroadcastType::NUMPY;
    case IE::AutoBroadcastType::PDPD:
        return ngraph::op::AutoBroadcastType::PDPD;
    default:
        VPUX_THROW("Unknown AutoBroadcastType");
    }
}

InferenceEngine::Precision NGraphExporter::exportPrecision(mlir::MLIRContext* ctx, mlir::Type type) {
    if (type == mlir::Float32Type::get(ctx)) {
        return InferenceEngine::Precision::FP32;
    } else if (type == mlir::Float16Type::get(ctx)) {
        return InferenceEngine::Precision::FP16;
    } else if (type == getSInt64Type(ctx)) {
        return InferenceEngine::Precision::I64;
    } else if (type == getUInt64Type(ctx)) {
        return InferenceEngine::Precision::U64;
    } else if (type == getSInt32Type(ctx)) {
        return InferenceEngine::Precision::I32;
    } else if (type == getUInt32Type(ctx)) {
        return InferenceEngine::Precision::U32;
    } else if (type == getSInt16Type(ctx)) {
        return InferenceEngine::Precision::I16;
    } else if (type == getUInt16Type(ctx)) {
        return InferenceEngine::Precision::U16;
    } else if (type == getSInt8Type(ctx)) {
        return InferenceEngine::Precision::I8;
    } else if (type == getUInt8Type(ctx)) {
        return InferenceEngine::Precision::U8;
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", type);
    }
}

ngraph::element::Type NGraphExporter::toNGraphType(InferenceEngine::Precision precision)
{
    if (precision == InferenceEngine::Precision::FP32)
        return ngraph::element::f32;
    else
        return ngraph::element::f32;

    if (precision == InferenceEngine::Precision::FP32) {
        return ngraph::element::f32;
    } else if (precision == InferenceEngine::Precision::FP16) {
        return ngraph::element::f16;
    } else if (precision == InferenceEngine::Precision::I64) {
        return ngraph::element::i64;
    } else if (precision == InferenceEngine::Precision::U64) {
        return ngraph::element::u64;
    } else if (precision == InferenceEngine::Precision::I32) {
        return ngraph::element::i32;
    } else if (precision == InferenceEngine::Precision::U32) {
        return ngraph::element::u32;
    } else if (precision == InferenceEngine::Precision::I16) {
        return ngraph::element::i16;
    } else if (precision == InferenceEngine::Precision::U16) {
        return ngraph::element::u16;
    } else if (precision == InferenceEngine::Precision::I8) {
        return ngraph::element::i8;
    } else if (precision == InferenceEngine::Precision::U8) {
        return ngraph::element::u8;
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", precision);
    }
}

}  // namespace

//
// exportToIRv10
//

void vpux::IE::exportToIRv10(mlir::ModuleOp module, std::ostringstream &ostr) {
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    NGraphExporter exporter;
    std::shared_ptr<ngraph::Function> netGraph = exporter.exportToNgraph(netOp, netFunc);
    InferenceEngine::CNNNetwork ieNet(netGraph);
    ieNet.serialize(ostr, ostr);
}

