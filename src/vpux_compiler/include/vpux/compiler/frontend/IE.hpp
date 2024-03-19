//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/Timing.h>

// Opset versions supported
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/opsets/opset9.hpp>

#include <ov_ops/nms_ie_internal.hpp>

// Utils
#include "vpux/utils/IE/hash.hpp"

namespace vpux {
namespace IE {

// TODO Get rid of this function (importNetwork), move logic to compiler.cpp
mlir::OwningOpRef<mlir::ModuleOp> importNetwork(mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model,
                                                bool sharedConstants, mlir::TimingScope& rootTiming,
                                                bool enableProfiling, bool stubLayers, vpux::VPU::ArchKind arch,
                                                Logger log = Logger::global());

// TODO Move to separate file NGraphPasses
class NGraphPasses final {
public:
    static void runNGraphPasses(const std::shared_ptr<ov::Model>& netGraph, mlir::TimingScope& rootTiming,
                                const vpux::VPU::ArchKind arch);
};

// TODO This variable is not tracked. Opset 7 number when we are supporting opset 11 on OV side
namespace opset_latest = ov::opset7;

class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ov::Model> netGraph, bool sharedConstants, Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

    mlir::func::FuncOp buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName, mlir::TimingScope& rootTiming,
                                     bool stubLayers);
    void buildBlockFromRegion(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block);
    SmallVector<mlir::Type> getRegionResults();
    static bool isOpSupported(const std::shared_ptr<ov::Node>& op);

private:
    using OrigNode = ov::Node;
    using OrigNodePtr = std::shared_ptr<OrigNode>;
    using NodeOutputMap = std::unordered_map<ov::Output<OrigNode>, mlir::Value>;
    using Callback = void (NGraphImporter::*)(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    static Callback getParser(const std::shared_ptr<ov::Node>& op);

    template <class NodeType>
    void parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
    }

    void parseNodeAsStub(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Constant>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convert>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Softmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogSoftmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tile>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Relu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Split>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Power>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Multiply>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Convolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GroupConvolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder,
                   const std::shared_ptr<opset_latest::GroupConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::AvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveAvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveMaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ShuffleChannels>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Gather>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::GatherND>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GatherTree>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::RandomUniform>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::OneHot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::BatchNormInference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GatherElements>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ScatterNDUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ScatterUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ScatterElementsUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Clamp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Elu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Reshape>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Squeeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMean>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceLogicalOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceLogicalAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceProd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceL1>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Unsqueeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Minimum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Maximum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Add>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Divide>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SquaredDifference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FloorMod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Mod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Proposal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FakeQuantize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MatMul>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tan>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Cos>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sqrt>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Cosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Log>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Selu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::Gelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Exp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HSwish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Floor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Round>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Mish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Erf>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Broadcast>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Bucketize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Transpose>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Interpolate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::RegionYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReorgYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DetectionOutput>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::NormalizeL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::CumSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::Eye>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::MVN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::MVN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Concat>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::PSROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v9::ROIAlign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::StridedSlice>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::PRelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Swish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Negative>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::CTCGreedyDecoder>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::CTCGreedyDecoderSeqLen>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Pad>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LSTMCell>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Subtract>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LSTMSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Ceiling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Equal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Select>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::NonMaxSuppression>& origNode);
    void parseNode(mlir::OpBuilder& builder,
                   const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::DepthToSpace>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReverseSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Less>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LessEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::NotEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SoftPlus>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Greater>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GreaterEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalNot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalXor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SpaceToDepth>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SpaceToBatch>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ExtractImagePatches>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Abs>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atan>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acos>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Roll>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HSigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HardSigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::GridSample>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::EmbeddingBagOffsetsSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::EmbeddingSegmentsSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::EmbeddingBagPackedSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Assign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ReadValue>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::Assign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::ReadValue>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRUCell>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GRUSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::DeformablePSROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::DFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::RDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::IDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::IRDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::If>& origNode);

    SmallVector<mlir::Value> getInputs(const OrigNodePtr& node);
    void addOutputs(const OrigNodePtr& node, mlir::Operation* op);
    mlir::Location createLocation(const OrigNodePtr& node);

    static SmallVector<int64_t> importShape(const ov::PartialShape& shape);
    mlir::Type importElemType(const ov::element::Type& elemType);
    mlir::RankedTensorType importTensor(const ov::PartialShape& shape, const ov::element::Type& elemType);
    IE::AutoBroadcastTypeAttr importBroadcastType(ov::op::AutoBroadcastType bType);
    IE::BroadcastTypeAttr importBroadcastMode(ov::op::BroadcastType bType);
    IE::RoundingTypeAttr importRoundingType(ov::op::RoundingType roundingType);
    IE::EpsModeAttr importEpsMode(ov::op::EpsMode val);
    IE::MvnEpsModeAttr importMvnEpsMode(ov::op::MVNEpsMode val);
    IE::TopKModeAttr importTopKMode(ov::op::TopKMode val);
    IE::TopKSortTypeAttr importTopKSortType(ov::op::TopKSortType val);
    IE::GridSampleModeAttr importGridSampleMode(const ov::op::v9::GridSample::InterpolationMode& val);
    IE::GridSamplePaddingModeAttr importGridSamplePaddingMode(const ov::op::v9::GridSample::PaddingMode& val);
    IE::ProposalAttr importProposalAttrs(const ov::op::v0::Proposal::Attributes& val);
    IE::InterpolateAttr importInterpolateAttrs(const opset_latest::Interpolate::InterpolateAttrs& val);
    IE::DetectionOutputAttr importDetectionOutputAttrs(const ov::op::v0::DetectionOutput::Attributes& val);
    IE::ROIPoolingMethodAttr importROIPoolingMethod(const std::string& method);
    IE::PSROIPoolingModeAttr importPSROIPoolingMode(const std::string& mode);
    IE::ROIAlignMethodAttr importROIAlignMethod(const ov::op::v9::ROIAlign::PoolingMode& mode);
    IE::ROIAlignAlignedMethodAttr importROIAlignAlignedMethod(const ov::op::v9::ROIAlign::AlignedMode& mode);
    IE::PadModeAttr importPadMode(const ov::op::PadMode val);
    IE::RoundModeAttr importRoundMode(const ov::op::v5::Round::RoundMode val);
    IE::RNNSequenceDirectionAttr importRNNSequenceDirection(const ov::op::RecurrentSequenceDirection val);
    IE::BoxEncodingTypeAttr importBoxEncodingType(const int val);
    IE::DepthToSpaceModeAttr importDepthToSpaceMode(const ov::op::v0::DepthToSpace::DepthToSpaceMode val);
    IE::SpaceToDepthModeAttr importSpaceToDepthMode(const ov::op::v0::SpaceToDepth::SpaceToDepthMode val);
    IE::PadTypeAttr importPadType(ov::op::PadType autoPads);
    IE::DeformablePSROIPoolingModeAttr importDeformablePSROIPoolingMode(const std::string& mode);
    IE::DetectionOutputCodeTypeAttr importDetectionOutputCodeType(const std::string& codeType);
    mlir::MLIRContext* _ctx = nullptr;
    std::shared_ptr<const ov::Model> _netGraph;
    bool _sharedConstants = false;
    Logger _log;

    NodeOutputMap _importedVals;
};

template <class NodeType>
void NGraphImporter::parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    auto targetPtr = std::dynamic_pointer_cast<NodeType>(origNode);
    OPENVINO_ASSERT(targetPtr != nullptr);
    parseNode(builder, targetPtr);
}

}  // namespace IE
}  // namespace vpux
