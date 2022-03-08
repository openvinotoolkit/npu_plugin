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
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"
#include "vpux/compiler/utils/attributes.hpp"
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
#include <llvm/ADT/DenseMap.h>

#include <ie_common.h>
#include <ie_layouts.h>
#include <ie_precision.hpp>

#include <cpp/ie_cnn_network.h>

#include <precision_utils.h>

#include <legacy/ngraph_ops/lrn_ie.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/function.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/node.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/type.hpp>

using namespace vpux;

std::shared_ptr<ngraph::Node> parseDeclareOp(Const::DeclareOp origOp)
{
    const auto& cont = origOp.content();
    const auto elType = cont.getElementType();
    auto ctx = elType.getContext();
    const auto elShape = cont.getShape();
    const ngraph::Shape sh(elShape.begin(), elShape.end());
    std::vector<int8_t> vals(cont.getNumElements() * cont.getElemTypeSize().count() / 8, 0);
    auto buffer = makeMutableArrayRef(reinterpret_cast<char*>(vals.data()), vals.size());
    cont.copyTo(buffer);
    return std::make_shared<opset_latest::Constant>(IE::exportElemType(ctx, elType), sh, buffer.data());
}

std::shared_ptr<ngraph::Node> parseLSTMSequenceOp(mlir::Operation *op, ngraph::OutputVector &outputs)
{
    auto lstmOp = mlir::dyn_cast<IE::LSTMSequenceOp>(*op);
    VPUX_THROW_UNLESS(lstmOp != nullptr, "Op is not of IE::LSTMSequenceOp type");
    auto seqLen = lstmOp.sequenceLength();
    auto elType = lstmOp.sequenceLengthAttr().getType();
    mlir::MLIRContext* ctx = elType.getContext();
    auto seqLenConstNode = std::make_shared<ov::op::v0::Constant>(IE::exportElemType(ctx, elType), ov::Shape{1}, static_cast<void*>(&seqLen));
    outputs.insert(outputs.cbegin() + 3, ov::Output<ov::Node>(seqLenConstNode, 0));
    auto opIface = mlir::dyn_cast<vpux::IE::ToNgraphOpInterface>(*op);
    VPUX_THROW_UNLESS(opIface != nullptr, "Op does not implement IE::ToNgraphOpInterface");
    return opIface.toNgraph(outputs);
}

std::shared_ptr<ngraph::Function> vpux::IE::exportToNgraph(IE::CNNNetworkOp, mlir::FuncOp netFunc)
{
    using NodeOutputMap = llvm::DenseMap<mlir::Value, ngraph::Output<ngraph::Node>>;

    ngraph::ParameterVector params;
    NodeOutputMap exportedVals;
    for (const auto& arg : netFunc.getArguments()) {
        auto type = arg.getType();
        auto rankedTensorType = type.dyn_cast<mlir::RankedTensorType>();
        VPUX_THROW_UNLESS(rankedTensorType != nullptr, "Value is not of mlir::RankedTensorType type");
        auto tensor = IE::exportUserTensor(rankedTensorType);
        ngraph::Shape shape{tensor.getDims().begin(), tensor.getDims().end()};
        auto param = std::make_shared<opset_latest::Parameter>(IE::toNGraphType(tensor.getPrecision()), shape);
        exportedVals.try_emplace(arg, param->get_default_output());
        params.push_back(param);
    }

    ngraph::OutputVector netOutputs;
    netFunc.body().walk([&](mlir::Operation *op) {
        ngraph::OutputVector inputs;
        for (auto val : op->getOperands()) {
            auto it = exportedVals.find(val);
            VPUX_THROW_WHEN(it == exportedVals.end(), "Value not found in mlir to ngraph translation map");
            inputs.push_back(it->second);
        }
        std::shared_ptr<ngraph::Node> node;
        if (auto constOp = mlir::dyn_cast<Const::DeclareOp>(*op)) {
            VPUX_THROW_UNLESS(constOp != nullptr, "Op {0} is not of Const::DeclareOp type", op->getName());
            node = parseDeclareOp(constOp);
            
        }
        else if (mlir::isa<IE::LSTMSequenceOp>(*op)) {
            node = parseLSTMSequenceOp(op, inputs);
        }
        else if (mlir::isa<mlir::ReturnOp>(*op)) {
            for (auto& input : inputs) {
                auto node = std::make_shared<opset_latest::Result>(input);
                netOutputs.emplace_back(node);
            }
        }
        else {
            auto opIface = mlir::dyn_cast<vpux::IE::ToNgraphOpInterface>(*op);
            VPUX_THROW_UNLESS(opIface != nullptr, "Op {0} does not implement IE::ToNgraphOpInterface", op->getName());
            node = opIface.toNgraph(inputs);
        }
        for (unsigned idx = 0; idx < op->getNumResults(); ++idx) {
            exportedVals.try_emplace(op->getResult(idx), ngraph::Output<ngraph::Node>(node, idx));
        }
    });

    return std::make_shared<ngraph::Function>(netOutputs, params);
}

//
// exportToOpenVINO
//
mlir::LogicalResult vpux::IE::exportToOpenVINO(mlir::ModuleOp module, llvm::raw_ostream&, const llvm::StringRef filePath) {
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    std::shared_ptr<ngraph::Function> netGraph = exportToNgraph(netOp, netFunc);
    InferenceEngine::CNNNetwork ieNet(netGraph);

    ieNet.serialize(filePath.str());

    return mlir::success();
}
