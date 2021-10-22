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
#include <legacy/ngraph_ops/lrn_ie.hpp>
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
#include <fstream>

using namespace vpux;

namespace {

    using OrigOp = mlir::Operation;
    using NodeMap = std::unordered_map<mlir::Operation *, std::shared_ptr<ngraph::Node>>;
class NGraphExporter final {
public:
    NGraphExporter() = default;

public:
    ngraph::element::Type exportElemType(mlir::MLIRContext* ctx, mlir::Type type);
    ngraph::op::AutoBroadcastType exportBroadcastType(IE::AutoBroadcastType bType);
    InferenceEngine::Precision exportPrecision(mlir::MLIRContext* ctx, mlir::Type type);
    InferenceEngine::TensorDesc exportUserTensor(llvm::SmallVector<IE::DataInfoOp> inputsInfo);
    ngraph::op::RoundingType exportRoundingType(IE::RoundingType roundingType);
    ngraph::element::Type toNGraphType(InferenceEngine::Precision precision);
    ngraph::op::RecurrentSequenceDirection exportRNNSequenceDirection(
        const IE::RNNSequenceDirection val);
    ngraph::op::PadMode exportPadMode(IE::PadMode mode);
    std::shared_ptr<ngraph::Function> exportToNgraph(IE::CNNNetworkOp netOp, mlir::FuncOp netFunc);
private:
    std::shared_ptr<ngraph::Function> _netGraph;
    std::shared_ptr<ngraph::Node> parseNode(Const::DeclareOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::AddOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::ConcatOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::ConvertOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::SoftMaxOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::TileOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::SplitOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::PowerOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::MultiplyOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::ReLUOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::ConvolutionOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::GroupConvolutionOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::AvgPoolOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::MaxPoolOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::GatherOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::ClampOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::EluOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::ReshapeOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::SqueezeOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::SigmoidOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::LRNOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::LRN_IEOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::FakeQuantizeOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::RegionYoloOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::PReluOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::NegativeOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::PadOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::LSTMCellOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::SubtractOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::AndOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(IE::LSTMSequenceOp origOp, ngraph::OutputVector &inputs);
    std::shared_ptr<ngraph::Node> parseNode(mlir::ReturnOp origOp, ngraph::OutputVector &inputs);
    template <class NodeType>
    std::shared_ptr<ngraph::Node> parseDispatch(OrigOp *origOp, ngraph::OutputVector &inputs) {
        return parseNode(llvm::dyn_cast<NodeType>(*origOp), inputs);
    }

    void parseEmpty(OrigOp *) {
    }
public:
    int _inputIndex = 0;
    NodeMap _importedVals;
    ngraph::ParameterVector _params;
};

std::shared_ptr<ngraph::Function> NGraphExporter::exportToNgraph(IE::CNNNetworkOp netOp, mlir::FuncOp netFunc)
{
    using Callback = std::shared_ptr<ngraph::Node> (NGraphExporter::*)(OrigOp *origOp, ngraph::OutputVector &inputs);
    using DispatchMap = std::map<std::string, Callback>;

#define MAP_ENTRY(_OpName_, _OpType_) \
    { _OpName_, &NGraphExporter::parseDispatch<_OpType_> }

    static DispatchMap dispatchMap {
            MAP_ENTRY("const.Declare", Const::DeclareOp),
            MAP_ENTRY("IE.Add", IE::AddOp),
            MAP_ENTRY("IE.Concat", IE::ConcatOp),
            MAP_ENTRY("IE.Convert", IE::ConvertOp),
            MAP_ENTRY("IE.SoftMax", IE::SoftMaxOp),
            MAP_ENTRY("IE.Tile", IE::TileOp),
            MAP_ENTRY("IE.Split", IE::SplitOp),
            MAP_ENTRY("IE.Power", IE::PowerOp),
            MAP_ENTRY("IE.Multiply", IE::MultiplyOp),
            MAP_ENTRY("IE.ReLU", IE::ReLUOp),
            MAP_ENTRY("IE.Convolution", IE::ConvolutionOp),
            MAP_ENTRY("IE.GroupConvolution", IE::GroupConvolutionOp),
            MAP_ENTRY("IE.AvgPool", IE::AvgPoolOp),
            MAP_ENTRY("IE.MaxPool", IE::MaxPoolOp),
            MAP_ENTRY("IE.Gather", IE::GatherOp),
            MAP_ENTRY("IE.Clamp", IE::ClampOp),
            MAP_ENTRY("IE.Elu", IE::EluOp),
            MAP_ENTRY("IE.Reshape", IE::ReshapeOp),
            MAP_ENTRY("IE.Squeeze", IE::SqueezeOp),
            MAP_ENTRY("IE.Sigmoid", IE::SigmoidOp),
            MAP_ENTRY("IE.LRN", IE::LRNOp),
            MAP_ENTRY("IE.FakeQuantize", IE::FakeQuantizeOp),
            MAP_ENTRY("IE.RegionYolo", IE::RegionYoloOp),
            MAP_ENTRY("IE.PRelu", IE::PReluOp),
            MAP_ENTRY("IE.Negative", IE::NegativeOp),
            MAP_ENTRY("IE.Pad", IE::PadOp),
            MAP_ENTRY("IE.LSTMCell", IE::LSTMCellOp),
            MAP_ENTRY("IE.Subtract", IE::SubtractOp),
            MAP_ENTRY("IE.And", IE::AndOp),
            MAP_ENTRY("IE.LSTMSequence", IE::LSTMSequenceOp),
            MAP_ENTRY("std.return", mlir::ReturnOp),
    };
#undef MAP_ENTRY

    llvm::raw_os_ostream os(std::cout);
    mlir::Block &block = *(netFunc.body().getBlocks().begin());
    block.walk([&](mlir::Operation *op) {
        os << "visiting op: '" << op->getName() << "' with "
            << op->getNumOperands() << " operands and "
            << op->getNumResults() << " results\n";
        os.flush();
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
std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(Const::DeclareOp origOp, ngraph::OutputVector &)
{
    auto cont = origOp.content();
    mlir::Type elType = cont.getElementType();
    mlir::MLIRContext* ctx = elType.getContext();
    auto valsRange = cont.getValues<double>();
    auto elShape = cont.getShape();
    ngraph::Shape sh(elShape.begin(), elShape.end());
    return std::make_shared<ngraph::opset7::Constant>(exportElemType(ctx, elType), sh, std::vector<double>(valsRange.begin(), valsRange.end()));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::AddOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::Add>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::ConcatOp origOp, ngraph::OutputVector &inputs)
{
    auto axis = origOp.axis();
    return std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{inputs.at(0), inputs.at(1)}, axis);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::ConvertOp origOp, ngraph::OutputVector &inputs)
{
    mlir::Type dstElemType = origOp.dstElemType();
    return std::make_shared<ngraph::opset7::Convert>(inputs.at(0), exportElemType(dstElemType.getContext(), dstElemType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::SoftMaxOp origOp, ngraph::OutputVector &inputs)
{
    auto axisIndVal = origOp.axisInd();
    return std::make_shared<ngraph::opset7::Softmax>(inputs.at(0), axisIndVal);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::TileOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Tile>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::SplitOp origOp, ngraph::OutputVector &inputs)
{
    auto numSplits = origOp.num_splits();
    return std::make_shared<ngraph::opset7::Split>(inputs.at(0), inputs.at(1), numSplits);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::PowerOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::Power>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::MultiplyOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::Multiply>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::ReLUOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Relu>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::ConvolutionOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_end());
    const auto dilations = parseIntArrayAttr<size_t>(origOp.dilations());
    return std::make_shared<ngraph::opset7::Convolution>(inputs.at(0), inputs.at(1), ngraph::Strides(strides.begin(),strides.end()),
        ngraph::CoordinateDiff(pads_begin.begin(), pads_begin.end()), ngraph::CoordinateDiff(pads_end.begin(), pads_end.end()),
        ngraph::Strides(dilations.begin(), dilations.end()), ngraph::op::PadType::SAME_UPPER);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::GroupConvolutionOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_end());
    const auto dilations = parseIntArrayAttr<size_t>(origOp.dilations());
    return std::make_shared<ngraph::opset7::GroupConvolution>(inputs.at(0), inputs.at(1), ngraph::Strides(strides.begin(),strides.end()),
        ngraph::CoordinateDiff(pads_begin.begin(), pads_begin.end()), ngraph::CoordinateDiff(pads_end.begin(), pads_end.end()),
        ngraph::Strides(dilations.begin(), dilations.end()), ngraph::op::PadType::SAME_UPPER);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::AvgPoolOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<size_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<size_t>(origOp.pads_end());
    const auto kernel = parseIntArrayAttr<size_t>(origOp.kernel_size());
    const auto exclude_pads = origOp.exclude_pads();
    const auto rounding_type = exportRoundingType(origOp.rounding_type());
    return std::make_shared<ngraph::opset7::AvgPool>(inputs.at(0), ngraph::Strides(strides.begin(),strides.end()),
        ngraph::Shape(pads_begin.begin(), pads_begin.end()), ngraph::Shape(pads_end.begin(), pads_end.end()),
        ngraph::Shape(kernel.begin(), kernel.end()), exclude_pads, rounding_type);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::MaxPoolOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<size_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<size_t>(origOp.pads_end());
    const auto kernel = parseIntArrayAttr<size_t>(origOp.kernel_size());
    const auto rounding_type = exportRoundingType(origOp.rounding_type());
    return std::make_shared<ngraph::opset7::MaxPool>(inputs.at(0), ngraph::Strides(strides.begin(), strides.end()),
        ngraph::Shape(pads_begin.begin(), pads_begin.end()), ngraph::Shape(pads_end.begin(), pads_end.end()),
        ngraph::Shape(kernel.begin(), kernel.end()), rounding_type);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::GatherOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Gather>(inputs.at(0), inputs.at(1), inputs.at(2));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::ClampOp origOp, ngraph::OutputVector &inputs)
{
    auto min = origOp.min();
    auto max = origOp.max();
    return std::make_shared<ngraph::opset7::Clamp>(inputs.at(0), min.convertToDouble(), max.convertToDouble());
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::EluOp origOp, ngraph::OutputVector &inputs)
{
    auto x = origOp.x();
    return std::make_shared<ngraph::opset7::Elu>(inputs.at(0), x.convertToDouble());
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::ReshapeOp origOp, ngraph::OutputVector &inputs)
{
    auto special_zero = origOp.special_zero();
    return std::make_shared<ngraph::opset7::Reshape>(inputs.at(0), inputs.at(1), special_zero);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::SqueezeOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Squeeze>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::SigmoidOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Sigmoid>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::LRNOp origOp, ngraph::OutputVector &inputs)
{
    auto alpha = origOp.alpha().convertToDouble();
    auto beta = origOp.beta().convertToDouble();
    auto bias = origOp.bias().convertToDouble();
    auto size = origOp.size();
    return std::make_shared<ngraph::opset7::LRN>(inputs.at(0), inputs.at(1), alpha, beta, bias, size);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::FakeQuantizeOp origOp, ngraph::OutputVector &inputs)
{
    auto levels = origOp.levels();
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::FakeQuantize>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        inputs.at(4), levels, ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::RegionYoloOp origOp, ngraph::OutputVector &inputs)
{
    const auto coords = origOp.coords();
    const auto classes = origOp.classes();
    const auto regions = origOp.regions();
    const auto do_softmax = origOp.do_softmax();
    const auto mask = parseIntArrayAttr<int64_t>(origOp.mask());
    const auto axis = origOp.axis();
    const auto end_axis = origOp.end_axis();
    const auto anchors = parseFPArrayAttr<float>(origOp.anchors());

    return std::make_shared<ngraph::opset7::RegionYolo>(inputs.at(0), coords, classes, regions, do_softmax,
        std::vector<int64_t>{mask.begin(),mask.end()}, axis, end_axis, std::vector<float>{anchors.begin(),anchors.end()});
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::PReluOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::PRelu>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::NegativeOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Negative>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::PadOp origOp, ngraph::OutputVector &inputs)
{
    const auto pad_mode = origOp.mode();
    if (inputs.size() == 4)
        return std::make_shared<ngraph::opset7::Pad>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        exportPadMode(pad_mode));
    else if (inputs.size() == 3)
        return std::make_shared<ngraph::opset7::Pad>(inputs.at(0), inputs.at(1), inputs.at(2),
        exportPadMode(pad_mode));
    else
        VPUX_THROW("IE::PadOp has unsupported number of inputs '{0}'", inputs.size());
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::LSTMCellOp origOp, ngraph::OutputVector &inputs)
{
    const auto hidden_size = origOp.hiddenSize();
    return std::make_shared<ngraph::opset7::LSTMCell>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        inputs.at(4), inputs.at(5), hidden_size);
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::SubtractOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::Subtract>(inputs.at(0), inputs.at(1),
        ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::LSTMSequenceOp origOp, ngraph::OutputVector &inputs)
{
    const auto hidden_size = origOp.sequenceLength();
    const auto lstm_direction = origOp.direction();

    return std::make_shared<ngraph::opset7::LSTMSequence>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        inputs.at(4), inputs.at(5), inputs.at(6), hidden_size, exportRNNSequenceDirection(lstm_direction));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(IE::AndOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::LogicalAnd>(inputs.at(0), inputs.at(1),
        ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> NGraphExporter::parseNode(mlir::ReturnOp, ngraph::OutputVector &inputs)
{
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

ngraph::element::Type NGraphExporter::exportElemType(mlir::MLIRContext* ctx, mlir::Type type) {
    if (type == mlir::Float32Type::get(ctx)) {
        return ngraph::element::f32;
    } else if (type == mlir::Float16Type::get(ctx)) {
        return ngraph::element::f16;
    } else if (type == getSInt64Type(ctx)) {
        return ngraph::element::i64;
    } else if (type == getUInt64Type(ctx)) {
        return ngraph::element::u64;
    } else if (type == getSInt32Type(ctx)) {
        return ngraph::element::i32;
    } else if (type == getUInt32Type(ctx)) {
        return ngraph::element::u32;
    } else if (type == getSInt16Type(ctx)) {
        return ngraph::element::i16;
    } else if (type == getUInt16Type(ctx)) {
        return ngraph::element::u16;
    } else if (type == getSInt8Type(ctx)) {
        return ngraph::element::i8;
    } else if (type == getUInt8Type(ctx)) {
        return ngraph::element::u8;
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

ngraph::op::RoundingType NGraphExporter::exportRoundingType(IE::RoundingType roundingType) {
    switch (roundingType) {
    case IE::RoundingType::FLOOR:
        return ngraph::op::RoundingType::FLOOR;
    case IE::RoundingType::CEIL:
        return ngraph::op::RoundingType::CEIL;
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}

ngraph::op::RecurrentSequenceDirection NGraphExporter::exportRNNSequenceDirection(
        const IE::RNNSequenceDirection val) {
    if (val == IE::RNNSequenceDirection::FORWARD) {
        return ngraph::op::RecurrentSequenceDirection::FORWARD;
    } else if (val == IE::RNNSequenceDirection::REVERSE) {
        return ngraph::op::RecurrentSequenceDirection::REVERSE;
    } else {
        VPUX_THROW("Unknown RNNSequence direction");
    }
}

ngraph::op::PadMode NGraphExporter::exportPadMode(IE::PadMode mode) {
    switch (mode) {
    case IE::PadMode::CONSTANT:
        return ngraph::op::PadMode::CONSTANT;
    case IE::PadMode::EDGE:
        return ngraph::op::PadMode::EDGE;
    case IE::PadMode::REFLECT:
        return ngraph::op::PadMode::REFLECT;
    case IE::PadMode::SYMMETRIC:
        return ngraph::op::PadMode::SYMMETRIC;
    default:
        VPUX_THROW("Unknown PadMode");
    }
}

}  // namespace

std::string provide_bin_path(const std::string &xmlPath) {
    assert(xmlPath.size() > 4); // should be check by valid_xml_path
    std::string bestPath = xmlPath;
    const char *const extension = "bin";
    const auto ext_size = std::strlen(extension);
    bestPath.replace(bestPath.size() - ext_size, ext_size, extension);
    return bestPath;
}

//
// exportToIRv10
//

mlir::LogicalResult vpux::IE::exportToIRv10(mlir::ModuleOp module, llvm::raw_ostream& output, const std::string &filePath) {
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    NGraphExporter exporter;
    std::shared_ptr<ngraph::Function> netGraph = exporter.exportToNgraph(netOp, netFunc);
    InferenceEngine::CNNNetwork ieNet(netGraph);

    std::ofstream binFile(provide_bin_path(filePath), std::ios::out | std::ios::binary);
    if (!binFile)
        return mlir::failure();
    std::ostringstream ostr;
    ieNet.serialize(ostr, binFile);
    output << ostr.str();
    return mlir::success();
}
