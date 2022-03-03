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

#include "vpux/passes/align_scales.hpp"
#include "vpux/passes/clean_up_fq.hpp"
#include "vpux/passes/convert_extract_image_patches_to_reorg_vpu.hpp"
#include "vpux/passes/convert_variadic_split_to_strided_slice.hpp"
#include "vpux/passes/fuse_scale_in_previous_weights_fq.hpp"
#include "vpux/passes/fuse_scaleshift.hpp"
#include "vpux/passes/propagate_fq.hpp"
#include "vpux/passes/remove_NV12_conversion.hpp"
#include "vpux/passes/remove_split_concat.hpp"
#include "vpux/passes/replace_onnx_pattern_to_reorg.hpp"

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

#include <legacy/ngraph_ops/lrn_ie.hpp>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/type/element_type.hpp>
#include "legacy/ngraph_ops/normalize_ie.hpp"
#include "vpux/passes/convert_MVN6_to_MVN1.hpp"

#include <legacy/transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_normalizel2_to_normalize_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/op_conversions/convert_gather_downgrade.hpp>
#include <transformations/op_conversions/convert_gather_upgrade.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_maxpool_downgrade.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/einsum_decomposition.hpp>
#include <transformations/op_conversions/gather_normalize_negative_indices.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/utils/utils.hpp>

#include <algorithm>

using namespace vpux;

namespace {

namespace opset_latest = ngraph::opset7;

//
// Sort parameters/results
//

//
// Inputs/outputs information in OV 1.0 is stored in DataMap, so they are sorted by the names.
// Sort the nGraph Function parameters/results to match the order of blobs in the maps.
//

ngraph::ParameterVector sortParameters(const ngraph::ParameterVector& orig) {
    ngraph::ParameterVector out = orig;
    std::sort(out.begin(), out.end(), [](auto&& p1, auto&& p2) {
        return p1->get_friendly_name() < p2->get_friendly_name();
    });
    return out;
}

ngraph::ResultVector sortResults(const ngraph::ResultVector& orig) {
    ngraph::ResultVector out = orig;
    std::sort(out.begin(), out.end(), [](auto&& r1, auto&& r2) {
        const auto n1 = ngraph::op::util::get_ie_output_name(r1->input_value(0));
        const auto n2 = ngraph::op::util::get_ie_output_name(r2->input_value(0));
        return n1 < n2;
    });
    return out;
}

//
// NGraphImporter
//

class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ngraph::Function> netGraph, bool sharedConstants,
                   Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

    mlir::FuncOp buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName, mlir::TimingScope& rootTiming);
    static std::unordered_set<std::string> getSupportedOps(std::shared_ptr<const ngraph::Function> netGraph);

private:
    using OrigNode = ngraph::Node;
    using OrigNodePtr = std::shared_ptr<OrigNode>;
    using NodeOutputMap = std::unordered_map<ngraph::Output<OrigNode>, mlir::Value>;
    using Callback = void (NGraphImporter::*)(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    static Callback getParser(const std::shared_ptr<ngraph::Node>& op);

    template <class NodeType>
    void parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
    }

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
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ShuffleChannels>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Gather>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::NV12toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::NV12toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::I420toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::I420toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::GatherElements>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ScatterNDUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Clamp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Elu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Reshape>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Squeeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::op::LRN_IE>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMean>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Unsqueeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Minimum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Maximum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Add>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Divide>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::SquaredDifference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FloorMod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Proposal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FakeQuantize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MatMul>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Tanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sqrt>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Cosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Log>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset2::Gelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Exp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::HSwish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Floor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Round>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Mish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Erf>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Broadcast>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Transpose>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Interpolate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::RegionYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReorgYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::DetectionOutput>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::op::NormalizeIE>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::MVN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Concat>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ROIAlign>& origNode);
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
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Abs>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atan>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acos>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Roll>& origNode);

    SmallVector<mlir::Value> getInputs(const OrigNodePtr& node);
    void addOutputs(const OrigNodePtr& node, mlir::Operation* op);
    mlir::Location createLocation(const OrigNodePtr& node);

    static SmallVector<int64_t> importShape(const ngraph::PartialShape& shape);
    mlir::Type importElemType(const ngraph::element::Type& elemType);
    mlir::RankedTensorType importTensor(const ngraph::PartialShape& shape, const ngraph::element::Type& elemType);
    IE::AutoBroadcastTypeAttr importBroadcastType(ngraph::op::AutoBroadcastType bType);
    IE::BroadcastTypeAttr importBroadcastMode(ngraph::op::BroadcastType bType);
    IE::RoundingTypeAttr importRoundingType(ngraph::op::RoundingType roundingType);
    IE::TopKModeAttr importTopKMode(ngraph::op::TopKMode val);
    IE::TopKSortTypeAttr importTopKSortType(ngraph::op::TopKSortType val);
    IE::ProposalAttr importProposalAttrs(const ngraph::op::ProposalAttrs& val);
    IE::InterpolateAttr importInterpolateAttrs(const opset_latest::Interpolate::InterpolateAttrs& val);
    IE::DetectionOutputAttr importDetectionOutputAttrs(const ngraph::op::DetectionOutputAttrs& val);
    IE::ROIPoolingMethodAttr importROIPoolingMethod(const std::string& method);
    IE::ROIAlignMethodAttr importROIAlignMethod(const ngraph::op::v3::ROIAlign::PoolingMode& mode);
    IE::PadModeAttr importPadMode(const ngraph::op::PadMode val);
    IE::RoundModeAttr importRoundMode(const ngraph::op::v5::Round::RoundMode val);
    IE::LRN_IERegionAttr importLRN_IERegion(const std::string& region);
    IE::RNNSequenceDirectionAttr importRNNSequenceDirection(const ngraph::op::RecurrentSequenceDirection val);
    IE::DepthToSpaceModeAttr importDepthToSpaceMode(const ngraph::op::v0::DepthToSpace::DepthToSpaceMode val);
    IE::SpaceToDepthModeAttr importSpaceToDepthMode(const ngraph::op::SpaceToDepth::SpaceToDepthMode val);

    mlir::MLIRContext* _ctx = nullptr;
    std::shared_ptr<const ngraph::Function> _netGraph;
    bool _sharedConstants = false;
    Logger _log;

    NodeOutputMap _importedVals;
};

template <class NodeType>
void NGraphImporter::parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    parseNode(builder, std::dynamic_pointer_cast<NodeType>(origNode));
}

NGraphImporter::Callback NGraphImporter::getParser(const std::shared_ptr<ngraph::Node>& op) {
    using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

#define MAP_ENTRY(_NodeType_) \
    { _NodeType_::get_type_info_static(), &NGraphImporter::parseDispatch<_NodeType_> }

    static const DispatchMap dispatchMap{
            {ngraph::op::Parameter::get_type_info_static(), &NGraphImporter::parseEmpty},
            {ngraph::op::Result::get_type_info_static(), &NGraphImporter::parseEmpty},

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
            MAP_ENTRY(opset_latest::ShuffleChannels),
            MAP_ENTRY(opset_latest::Gather),
            MAP_ENTRY(ngraph::opset8::NV12toRGB),
            MAP_ENTRY(ngraph::opset8::NV12toBGR),
            MAP_ENTRY(ngraph::opset8::I420toRGB),
            MAP_ENTRY(ngraph::opset8::I420toBGR),
            MAP_ENTRY(opset_latest::GatherElements),
            MAP_ENTRY(opset_latest::ScatterNDUpdate),
            MAP_ENTRY(opset_latest::Clamp),
            MAP_ENTRY(opset_latest::Elu),
            MAP_ENTRY(opset_latest::Reshape),
            MAP_ENTRY(opset_latest::Squeeze),
            MAP_ENTRY(opset_latest::Sigmoid),
            MAP_ENTRY(opset_latest::LRN),
            MAP_ENTRY(ngraph::op::LRN_IE),
            MAP_ENTRY(opset_latest::ReduceMax),
            MAP_ENTRY(opset_latest::ReduceMean),
            MAP_ENTRY(opset_latest::ReduceSum),
            MAP_ENTRY(opset_latest::Unsqueeze),
            MAP_ENTRY(opset_latest::Minimum),
            MAP_ENTRY(opset_latest::Maximum),
            MAP_ENTRY(opset_latest::Add),
            MAP_ENTRY(opset_latest::Divide),
            MAP_ENTRY(opset_latest::SquaredDifference),
            MAP_ENTRY(opset_latest::FloorMod),
            MAP_ENTRY(opset_latest::Proposal),
            MAP_ENTRY(opset_latest::FakeQuantize),
            MAP_ENTRY(opset_latest::MatMul),
            MAP_ENTRY(opset_latest::Tanh),
            MAP_ENTRY(opset_latest::Sqrt),
            MAP_ENTRY(opset_latest::Sinh),
            MAP_ENTRY(opset_latest::Cosh),
            MAP_ENTRY(opset_latest::Asinh),
            MAP_ENTRY(opset_latest::Acosh),
            MAP_ENTRY(opset_latest::Atanh),
            MAP_ENTRY(opset_latest::Log),
            MAP_ENTRY(ngraph::opset2::Gelu),
            MAP_ENTRY(opset_latest::Exp),
            MAP_ENTRY(opset_latest::HSwish),
            MAP_ENTRY(opset_latest::Floor),
            MAP_ENTRY(opset_latest::Round),
            MAP_ENTRY(opset_latest::Mish),
            MAP_ENTRY(opset_latest::Erf),
            MAP_ENTRY(opset_latest::Broadcast),
            MAP_ENTRY(opset_latest::Transpose),
            MAP_ENTRY(opset_latest::Interpolate),
            MAP_ENTRY(opset_latest::TopK),
            MAP_ENTRY(opset_latest::RegionYolo),
            MAP_ENTRY(opset_latest::ReorgYolo),
            MAP_ENTRY(opset_latest::DetectionOutput),
            MAP_ENTRY(ngraph::op::NormalizeIE),
            MAP_ENTRY(ngraph::opset4::MVN),
            MAP_ENTRY(opset_latest::Concat),
            MAP_ENTRY(opset_latest::ROIPooling),
            MAP_ENTRY(opset_latest::ROIAlign),
            MAP_ENTRY(opset_latest::StridedSlice),
            MAP_ENTRY(opset_latest::PRelu),
            MAP_ENTRY(opset_latest::Swish),
            MAP_ENTRY(opset_latest::GRN),
            MAP_ENTRY(opset_latest::Negative),
            MAP_ENTRY(opset_latest::Sign),
            MAP_ENTRY(opset_latest::CTCGreedyDecoder),
            MAP_ENTRY(opset_latest::CTCGreedyDecoderSeqLen),
            MAP_ENTRY(opset_latest::Pad),
            MAP_ENTRY(opset_latest::LSTMCell),
            MAP_ENTRY(opset_latest::Subtract),
            MAP_ENTRY(opset_latest::LogicalAnd),
            MAP_ENTRY(opset_latest::LSTMSequence),
            MAP_ENTRY(opset_latest::Ceiling),
            MAP_ENTRY(opset_latest::SoftPlus),
            MAP_ENTRY(opset_latest::Equal),
            MAP_ENTRY(opset_latest::Select),
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
            MAP_ENTRY(opset_latest::Abs),
            MAP_ENTRY(opset_latest::Atan),
            MAP_ENTRY(opset_latest::Asin),
            MAP_ENTRY(opset_latest::Acos),
            MAP_ENTRY(opset_latest::Roll),
    };

#undef MAP_ENTRY

    const auto dispatchIt = dispatchMap.find(op->get_type_info());
    return (dispatchIt != dispatchMap.end()) ? dispatchIt->second : nullptr;
}

std::unordered_set<std::string> NGraphImporter::getSupportedOps(std::shared_ptr<const ngraph::Function> netGraph) {
    std::unordered_set<std::string> supported;
    std::unordered_set<std::string> unsupported;
    for (const auto& op : netGraph->get_ordered_ops()) {
        const bool hasParser = (getParser(op) != nullptr);
        for (auto&& fusedLayerName : ngraph::getFusedNamesVector(op)) {
            if (hasParser) {
                supported.emplace(fusedLayerName);
            } else {
                unsupported.emplace(fusedLayerName);
            }
        }
    }
    for (auto&& unsupportedNode : unsupported) {
        supported.erase(unsupportedNode);
    }
    return supported;
}

//
// buildMainFunc
//

mlir::FuncOp NGraphImporter::buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName,
                                           mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Import nGraph function");

    const auto sortedParameters = sortParameters(_netGraph->get_parameters());
    const auto sortedResults = sortResults(_netGraph->get_results());

    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(sortedParameters.size());
    for (const auto& param : sortedParameters) {
        inputTypes.push_back(importTensor(param->get_partial_shape(), param->get_element_type()));
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(sortedResults.size());
    for (const auto& result : sortedResults) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }

    const auto funcType = mlir::FunctionType::get(_ctx, makeArrayRef(inputTypes), makeArrayRef(outputTypes));

    auto func = moduleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(_ctx), funcName, funcType);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    for (const auto& p : sortedParameters | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = checked_cast<uint32_t>(p.index());

        _log.trace("Convert network Parameter {0}", paramNode->get_friendly_name());

        const auto funcInputVal = func.getArgument(paramIndex);
        _importedVals.emplace(paramNode->output(0), funcInputVal);
    }

    for (const auto& origNode : _netGraph->get_ordered_ops()) {
        _log.trace("Convert {0} layer {1}", origNode->get_type_name(), origNode->get_friendly_name());
        const auto parser = NGraphImporter::getParser(origNode);
        VPUX_THROW_UNLESS(nullptr != parser, "Unsupported operation {0} with type {1}", origNode->get_friendly_name(),
                          origNode->get_type_name());

        (this->*parser)(builder, origNode);
    }

    SmallVector<mlir::Value> funcOutputs;
    funcOutputs.reserve(sortedResults.size());

    for (const auto& p : sortedResults | indexed) {
        const auto& resultNode = p.value();

        _log.trace("Convert network Result {0}", resultNode->get_friendly_name());

        const auto resultInputs = getInputs(resultNode);
        VPUX_THROW_UNLESS(resultInputs.size() == 1, "nGraph Result {0} has unsupported number of inputs {1}",
                          resultNode->get_friendly_name(), resultInputs.size());

        funcOutputs.push_back(resultInputs[0]);
    }

    builder.create<mlir::ReturnOp>(mlir::NameLoc::get(mlir::Identifier::get("output", _ctx)),
                                   makeArrayRef(funcOutputs));

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
    const auto axisAttr = getIntAttr(_ctx, axis);

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
    const auto numSplitsAttr = getIntAttr(_ctx, num_splits);

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
                                             importBroadcastType(autob.m_type), nullptr);
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

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());

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

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::GroupConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr,
                                                     attrStride, attrPadsBegin, attrPadsEnd, attrDilation,
                                                     /*groups=*/nullptr,
                                                     /*post_op=*/nullptr);
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

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrOutputPadding = getIntArrayAttr(_ctx, origNode->get_output_padding());

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

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());
    const auto attrExcludePads = origNode->get_exclude_pad() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::AvgPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType, attrExcludePads);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::MaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::MaxPool>::value,
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

    auto op =
            builder.create<IE::AddOp>(createLocation(origNode), inputs[0], inputs[1], importBroadcastType(autob.m_type),
                                      /*post_op=*/nullptr);
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::ShuffleChannels>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::ShuffleChannels>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ShuffleChannels node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ShuffleChannelsOp>(createLocation(origNode), inputs[0], origNode->get_axis(),
                                                    origNode->get_group());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Gather>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v7::Gather>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::NV12toRGB>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::opset8::NV12toRGB>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::NV12toBGR>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::opset8::NV12toBGR>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::I420toRGB>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::opset8::I420toRGB>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset8::I420toBGR>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::opset8::I420toBGR>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<opset_latest::GatherElements>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v6::GatherElements>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v3::ScatterNDUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph ScatterNDUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ScatterNDUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
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
    const auto minAttr = getFPAttr(_ctx, min);
    const auto maxAttr = getFPAttr(_ctx, max);

    auto op = builder.create<IE::ClampOp>(createLocation(origNode), inputs[0], minAttr, maxAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Proposal>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::Proposal>::value,
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

    const auto alphaAttr = getFPAttr(_ctx, alpha);
    const auto betaAttr = getFPAttr(_ctx, beta);
    const auto biasAttr = getFPAttr(_ctx, bias);
    const auto sizeAttr = getIntAttr(_ctx, size);

    auto op = builder.create<IE::LRNOp>(createLocation(origNode), inputs[0], inputs[1], alphaAttr, betaAttr, biasAttr,
                                        sizeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::op::LRN_IE>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::LRN_IE>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph LRN_IE node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto beta = origNode->get_beta();
    const auto bias = origNode->get_bias();
    const auto size = origNode->get_nsize();

    const auto alphaAttr = getFPAttr(_ctx, alpha);
    const auto betaAttr = getFPAttr(_ctx, beta);
    const auto biasAttr = getFPAttr(_ctx, bias);
    const auto sizeAttr = getIntAttr(_ctx, size);
    const auto regionAttr = importLRN_IERegion(origNode->get_region());

    auto op = builder.create<IE::LRN_IEOp>(createLocation(origNode), inputs[0], alphaAttr, betaAttr, biasAttr, sizeAttr,
                                           regionAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Broadcast>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v3::Broadcast>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::ReduceMax>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMaxOp>(createLocation(origNode), inputs[0], inputs[1], keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceMean>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::ReduceMean>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMean node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMeanOp>(createLocation(origNode), inputs[0], inputs[1], keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ReduceSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::ReduceSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceSum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceSumOp>(createLocation(origNode), inputs[0], inputs[1], keep_dims);
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sqrt>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Sqrt>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sqrt node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SqrtOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sinh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Sinh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sinh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SinhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Cosh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Cosh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Cosh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CoshOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asinh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v3::Asinh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Asinh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AsinhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acosh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v3::Acosh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Acosh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AcoshOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atanh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v3::Atanh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Atanh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AtanhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Log>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Log>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Log node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::LogOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset2::Gelu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Gelu>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Gelu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::GeluOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Elu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Elu>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::HSwish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph HSwish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HSwishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Floor>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Floor>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Floor node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::FloorOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Round>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v5::Round>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Round node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RoundOp>(createLocation(origNode), inputs[0], importRoundMode(origNode->get_mode()));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Mish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::Mish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Mish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::MishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Erf>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Erf>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Erf node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ErfOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::FakeQuantize>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::FakeQuantize>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::ROIPooling>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::ROIAlign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v3::ROIAlign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph ROIAlign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto pooled_h = getIntAttr(_ctx, origNode->get_pooled_h());
    const auto pooled_w = getIntAttr(_ctx, origNode->get_pooled_w());
    const auto sampling_ratio = getIntAttr(_ctx, origNode->get_sampling_ratio());
    const auto spatialScaleAttr = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto poolingMode = importROIAlignMethod(origNode->get_mode());

    auto op = builder.create<IE::ROIAlignOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], pooled_h,
                                             pooled_w, sampling_ratio, spatialScaleAttr, poolingMode);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Concat>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Concat>::value,
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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::TopK>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v3::TopK>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph TopK node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
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

    const auto strideAttr = getIntAttr(_ctx, strides.front());

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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::op::NormalizeIE>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::NormalizeIE>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Normalize node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto epsAttr = getFPAttr(_ctx, origNode->get_eps());
    const auto across_spatialAttr = mlir::BoolAttr::get(_ctx, origNode->get_across_spatial());
    const auto channel_sharedAttr = mlir::BoolAttr::get(_ctx, origNode->get_channel_shared());

    auto op = builder.create<IE::NormalizeIEOp>(createLocation(origNode), inputs[0], inputs[1], epsAttr,
                                                across_spatialAttr, channel_sharedAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ngraph::opset4::MVN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::MVN>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::GRN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph GRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto biasAttr = getFPAttr(_ctx, origNode->get_bias());

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

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Sign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Sign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SignOp>(createLocation(origNode), inputs[0]);
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::LSTMCell>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Subtract>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SubtractOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type), nullptr);
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
                                        importBroadcastType(autob.m_type), nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Ceiling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Ceiling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Ceiling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CeilingOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Equal>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Equal>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Select>::value,
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
                               const std::shared_ptr<opset_latest::ReverseSequence>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::ReverseSequence>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v4::SoftPlus>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph SoftPlus node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SoftPlusOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Less>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Less>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::LessEqual>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::Greater>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::GreaterEqual>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::NotEqual>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::DepthToSpace>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::LogicalNot>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::LogicalNotOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::LogicalOr>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::LogicalOr>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v1::LogicalXor>::value,
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
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::SpaceToDepth>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph SpaceToDepth node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto blockSizeAttr = getIntAttr(_ctx, origNode->get_block_size());
    const auto modeAttr = importSpaceToDepthMode(origNode->get_mode());

    auto op = builder.create<IE::SpaceToDepthOp>(createLocation(origNode), inputs[0], blockSizeAttr, modeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Abs>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Abs>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Abs node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AbsOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Atan>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Atan>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Atan node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AtanOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Asin>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Asin>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Asin node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AsinOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Acos>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v0::Acos>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Acos node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AcosOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<opset_latest::Roll>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ngraph::op::v7::Roll>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Roll node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RollOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
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
    } else if (elemType == ngraph::element::boolean) {
        return getBool8Type(_ctx);
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

IE::BroadcastTypeAttr NGraphImporter::importBroadcastMode(ngraph::op::BroadcastType bType) {
    switch (bType) {
    case ngraph::op::BroadcastType::NUMPY:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::NUMPY);
    case ngraph::op::BroadcastType::EXPLICIT:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::EXPLICIT);
    case ngraph::op::BroadcastType::BIDIRECTIONAL:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::BIDIRECTIONAL);
    default:
        VPUX_THROW("Unknown BroadcastMode");
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

IE::LRN_IERegionAttr NGraphImporter::importLRN_IERegion(const std::string& region) {
    if (region == "same") {
        return IE::LRN_IERegionAttr::get(_ctx, IE::LRN_IERegion::same);
    } else if (region == "across") {
        return IE::LRN_IERegionAttr::get(_ctx, IE::LRN_IERegion::across);
    } else {
        VPUX_THROW("Unknown LRN_IERegion");
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

    return IE::ProposalAttr::get(baseSizeAttr, preNmsTopNAttr, postNmsTopNAttr, nmsThreshNAttr, featStrideAttr,
                                 minSizeNAttr, ratioAttr, scaleAttr, clipBeforeNmsAttr, clipAfterNmsAttr, normalizeAttr,
                                 boxSizeScaleAttr, boxCoordinateScaleAttr, frameworkAttr, inferProbsAttr, _ctx);
}

IE::InterpolateAttr NGraphImporter::importInterpolateAttrs(const opset_latest::Interpolate::InterpolateAttrs& val) {
    // mode
    IE::InterpolateModeAttr modeAttr;
    switch (val.mode) {
    case opset_latest::Interpolate::InterpolateMode::NEAREST:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::nearest);
        break;
    case opset_latest::Interpolate::InterpolateMode::LINEAR:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::linear);
        break;
    case opset_latest::Interpolate::InterpolateMode::LINEAR_ONNX:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::linear_onnx);
        break;
    case opset_latest::Interpolate::InterpolateMode::CUBIC:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::cubic);
        break;
    default:
        VPUX_THROW("Unsupported interpolate mode");
    }

    // shape calculation mode
    IE::InterpolateCalcModeAttr calcModeAttr;
    switch (val.shape_calculation_mode) {
    case opset_latest::Interpolate::ShapeCalcMode::SIZES:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::sizes);
        break;
    case opset_latest::Interpolate::ShapeCalcMode::SCALES:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::scales);
        break;
    default:
        VPUX_THROW("Unsupported interpolate shape calculation mode");
    }

    // coordinate transformation mode
    IE::InterpolateCoordModeAttr coordModeAttr;
    switch (val.coordinate_transformation_mode) {
    case opset_latest::Interpolate::CoordinateTransformMode::HALF_PIXEL:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::half_pixel);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::pytorch_half_pixel);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::ASYMMETRIC:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::asymmetric);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::tf_half_pixel_for_nn);
        break;
    case opset_latest::Interpolate::CoordinateTransformMode::ALIGN_CORNERS:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::align_corners);
        break;
    default:
        VPUX_THROW("Unsupported interpolate coordinate transformation mode");
    }

    // coordinate transformation mode
    IE::InterpolateNearestModeAttr nearestModeAttr;
    switch (val.nearest_mode) {
    case opset_latest::Interpolate::NearestMode::ROUND_PREFER_FLOOR:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::round_prefer_floor);
        break;
    case opset_latest::Interpolate::NearestMode::ROUND_PREFER_CEIL:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::round_prefer_ceil);
        break;
    case opset_latest::Interpolate::NearestMode::FLOOR:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::floor);
        break;
    case opset_latest::Interpolate::NearestMode::CEIL:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::ceil);
        break;
    case opset_latest::Interpolate::NearestMode::SIMPLE:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::simple);
        break;
    default:
        VPUX_THROW("Unsupported interpolate nearest mode");
    }

    const auto antialiasAttr = mlir::BoolAttr::get(_ctx, val.antialias);
    const auto padsBeginAttr = getIntArrayAttr(_ctx, val.pads_begin);
    const auto padsEndAttr = getIntArrayAttr(_ctx, val.pads_end);
    const auto cubeCoeffAttr = getFPAttr(_ctx, val.cube_coeff);

    return IE::InterpolateAttr::get(modeAttr, calcModeAttr, coordModeAttr, nearestModeAttr, antialiasAttr,
                                    padsBeginAttr, padsEndAttr, cubeCoeffAttr, _ctx);
}

IE::DetectionOutputAttr NGraphImporter::importDetectionOutputAttrs(const ngraph::op::DetectionOutputAttrs& val) {
    const auto numClassesAttr = getIntAttr(_ctx, val.num_classes);
    const auto backgroundLabelIdAttr = getIntAttr(_ctx, val.background_label_id);
    const auto topKAttr = getIntAttr(_ctx, val.top_k);

    const auto varianceEncodedInTargetAttr = mlir::BoolAttr::get(_ctx, val.variance_encoded_in_target);

    const auto keepTopKAttr = getIntArrayAttr(_ctx, val.keep_top_k);
    const auto codeTypeAttr = mlir::StringAttr::get(_ctx, val.code_type);

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

IE::ROIAlignMethodAttr NGraphImporter::importROIAlignMethod(const ngraph::op::v3::ROIAlign::PoolingMode& mode) {
    IE::ROIAlignMethodAttr attr;
    if (mode == ngraph::op::v3::ROIAlign::PoolingMode::AVG) {
        attr = IE::ROIAlignMethodAttr::get(_ctx, IE::ROIAlignMethod::avg);
    } else if (mode == ngraph::op::v3::ROIAlign::PoolingMode::MAX) {
        attr = IE::ROIAlignMethodAttr::get(_ctx, IE::ROIAlignMethod::max);
    } else {
        VPUX_THROW("Unknown ROIAlignMethod");
    }
    return attr;
}

IE::RNNSequenceDirectionAttr NGraphImporter::importRNNSequenceDirection(
        const ngraph::op::RecurrentSequenceDirection val) {
    IE::RNNSequenceDirectionAttr attr;
    if (val == ngraph::op::RecurrentSequenceDirection::FORWARD) {
        attr = IE::RNNSequenceDirectionAttr::get(_ctx, IE::RNNSequenceDirection::FORWARD);
    } else if (val == ngraph::op::RecurrentSequenceDirection::REVERSE) {
        attr = IE::RNNSequenceDirectionAttr::get(_ctx, IE::RNNSequenceDirection::REVERSE);
    } else {
        VPUX_THROW("Unknown RNNSequence direction");
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

IE::RoundModeAttr NGraphImporter::importRoundMode(const ngraph::op::v5::Round::RoundMode val) {
    IE::RoundModeAttr attr;
    switch (val) {
    case ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN:
        attr = IE::RoundModeAttr::get(_ctx, IE::RoundMode::HALF_TO_EVEN);
        break;
    case ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO:
        attr = IE::RoundModeAttr::get(_ctx, IE::RoundMode::HALF_AWAY_FROM_ZERO);
        break;
    default:
        VPUX_THROW("Unknown RoundMode");
    }
    return attr;
}

IE::DepthToSpaceModeAttr NGraphImporter::importDepthToSpaceMode(
        const ngraph::op::v0::DepthToSpace::DepthToSpaceMode val) {
    IE::DepthToSpaceModeAttr attr;
    switch (val) {
    case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
        attr = IE::DepthToSpaceModeAttr::get(_ctx, IE::DepthToSpaceMode::BLOCKS_FIRST);
        break;
    case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
        attr = IE::DepthToSpaceModeAttr::get(_ctx, IE::DepthToSpaceMode::DEPTH_FIRST);
        break;
    default:
        VPUX_THROW("Unknown DepthToSpace Mode");
    }
    return attr;
}

IE::SpaceToDepthModeAttr NGraphImporter::importSpaceToDepthMode(const ngraph::op::SpaceToDepth::SpaceToDepthMode val) {
    IE::SpaceToDepthModeAttr attr;
    switch (val) {
    case ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
        attr = IE::SpaceToDepthModeAttr::get(_ctx, IE::SpaceToDepthMode::BLOCKS_FIRST);
        break;
    case ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
        attr = IE::SpaceToDepthModeAttr::get(_ctx, IE::SpaceToDepthMode::DEPTH_FIRST);
        break;
    default:
        VPUX_THROW("Unknown SpaceToDepthMode");
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
    return getTensorType(shape, precision, DimsOrder::fromIE(desc.getLayout()), nullptr);
}

//
// runNGraphPasses
//

static void addCommonOptimizationsPasses(ngraph::pass::Manager& manager) {
    // Disable low_precision_enabled as all plugins handle low-precision sub-graph manually
    // before CommonOptimization pipeline execution
    manager.register_pass<ngraph::pass::MOCTransformations>(true, false);

    auto pass_config = manager.get_pass_config();
    pass_config->disable<ngraph::pass::PadFusionConvolution>();
    pass_config->disable<ngraph::pass::PadFusionGroupConvolution>();

    auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    common_fusions->add_matcher<ngraph::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ngraph::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ngraph::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ngraph::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ngraph::pass::TransposeToReshape>();
    common_fusions->set_name("ngraph::pass::CommonFusions");

    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::Gelu7Downgrade>();
    decomp->add_matcher<ngraph::pass::BidirectionalSequenceDecomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL1Decomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL2Decomposition>();
    decomp->add_matcher<ngraph::pass::LogSoftmaxDecomposition>();
    decomp->add_matcher<ngraph::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ngraph::pass::ConvertMod>();
    decomp->add_matcher<ngraph::pass::ConvertGELU>();
    decomp->add_matcher<ngraph::pass::BatchNormDecomposition>();
    decomp->add_matcher<ngraph::pass::EinsumDecomposition>();
    decomp->add_matcher<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    decomp->add_matcher<ngraph::pass::DropoutWithRandomUniformReplacer>();
    decomp->set_name("ngraph::pass::CommonDecompositions");

    // CF is required after all decompositions
    manager.register_pass<ngraph::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
  //  manager.register_pass<ngraph::pass::UnrollIf>();

    auto conv_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    conv_fusions->add_matcher<ngraph::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::MultiplyConvolutionFusion>();
    conv_fusions->add_matcher<ngraph::pass::MultiplyGroupConvolutionFusion>();
    conv_fusions->add_matcher<ngraph::pass::MultiplyConvolutionBackpropDataFusion>();
    conv_fusions->add_matcher<ngraph::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    conv_fusions->set_name("ngraph::pass::ConvFusions");

    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertGather8ToGather7>();  // not plugins implemented gather8
    manager.register_pass<ngraph::pass::ConvertGather7ToGather1>();  // not plugins implemented gather7
    manager.register_pass<ngraph::pass::ConvertGather1ToGather7>();
    manager.register_pass<ngraph::pass::ConvertDeformableConv8To1>();
    manager.register_pass<ngraph::pass::ConvertMaxPool8ToMaxPool1>();

    // StridesOptimization should be at the very end
    // because we cannot insert any MaxPools since they may prevent
    // other optimizations
    manager.register_pass<ngraph::pass::StridesOptimization>();
}

void runNGraphPasses(const std::shared_ptr<ngraph::Function>& netGraph,
                     std::vector<vpux::PreProcessInfo>& /*preProcInfo*/, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Common nGraph passes");

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<vpux::pass::RemoveSplitConcat>();
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<vpux::pass::FuseScaleShift>();
    manager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertGELU>();
    manager.register_pass<vpux::passes::OnnxReorgPatternToDarkNetReorg>();
    manager.register_pass<vpux::passes::ConvertExtractImagePatchesToReorgYoloVPU>();
    manager.register_pass<vpux::pass::FuseScaleAfterClamp>();
    addCommonOptimizationsPasses(manager);

    manager.register_pass<vpux::passes::PropagateFQ>();
    manager.register_pass<vpux::passes::AlignScales>();
    manager.register_pass<ngraph::pass::ReluFakeQuantizeFusion>();
    // we need additionally propagate FQs because some ReLUs may be removed
    manager.register_pass<vpux::passes::PropagateFQ>();
    manager.register_pass<vpux::passes::CleanUpFQ>();

    manager.register_pass<vpux::passes::ConvertMVN6toMVN1>();
    manager.register_pass<ngraph::pass::ConvertLRNToLegacyMatcher>();
    manager.register_pass<vpux::passes::ConvertVariadicSplitToStridedSliceOp>();
    manager.register_pass<ngraph::pass::ConvertNormalizeL2ToLegacyMatcher>();

    manager.run_passes(netGraph);
}

//
// addCNNNetworkOp
//

void addCNNNetworkOp(mlir::OpBuilder& builder, mlir::FlatSymbolRefAttr mainFuncName, InferenceEngine::CNNNetwork cnnNet,
                     const std::shared_ptr<ngraph::Function>& netGraph, mlir::TimingScope& rootTiming,
                     bool enableProfiling) {
    auto scopeTiming = rootTiming.nest("Add CNNNetwork Operation");

    const auto inputsInfo = cnnNet.getInputsInfo();
    const auto outputsInfo = cnnNet.getOutputsInfo();

    const auto sortedParameters = sortParameters(netGraph->get_parameters());
    const auto sortedResults = sortResults(netGraph->get_results());

    auto* ctx = builder.getContext();

    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(ctx), mainFuncName, enableProfiling);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();
    if (enableProfiling) {
        cnnOp.profilingOutputsInfo().front().emplaceBlock();
    }

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), builder.getListener());
    for (const auto& param : sortedParameters) {
        const auto& inputName = param->get_friendly_name();
        const auto& userInput = inputsInfo.at(inputName);
        const auto& userDesc = userInput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(ctx, inputName);
        const auto userTypeAttr = mlir::TypeAttr::get(importUserTensor(ctx, userDesc));

        inputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), builder.getListener());
    for (const auto& result : sortedResults) {
        const auto resultName = ngraph::op::util::get_ie_output_name(result->input_value(0));
        const auto& userOutput = outputsInfo.at(resultName);
        const auto& userDesc = userOutput->getTensorDesc();

        const auto nameAttr = mlir::StringAttr::get(ctx, resultName);
        const auto userTypeAttr = mlir::TypeAttr::get(importUserTensor(ctx, userDesc));

        outputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, userTypeAttr);
    }
}

//
// validateCNNNetwork
//

void validateCNNNetwork(const InferenceEngine::CNNNetwork& cnnNet) {
    const auto inputsInfo = cnnNet.getInputsInfo();

    for (const auto& p : inputsInfo) {
        const auto& name = p.first;
        const auto& info = p.second;
        const auto& preProc = info->getPreProcess();
        const auto meanVariant = preProc.getMeanVariant();
        VPUX_THROW_UNLESS(meanVariant == InferenceEngine::MeanVariant::NONE,
                          "MeanVariant pre-processing for input '{0}' is not supported", name);
    }
}

}  // namespace

//
// queryNetwork
//

std::unordered_set<std::string> vpux::IE::queryNetwork(const InferenceEngine::CNNNetwork& cnnNet,
                                                       std::vector<PreProcessInfo>& preProcInfo,
                                                       mlir::TimingScope& rootTiming, Logger log) {
    log.setName("IE::FrontEnd::queryNetwork");

    validateCNNNetwork(cnnNet);

    const auto netGraph = ngraph::clone_function(*(cnnNet.getFunction()));
    VPUX_THROW_UNLESS(netGraph != nullptr, "Old IR versions (prior v10) are not supported : {0}", cnnNet.getName());

    log.trace("Run common nGraph passes");
    runNGraphPasses(netGraph, preProcInfo, rootTiming);

    log.trace("Get supported operations list");
    return NGraphImporter::getSupportedOps(netGraph);
}

//
// importNetwork
//

mlir::OwningModuleRef vpux::IE::importNetwork(mlir::MLIRContext* ctx, InferenceEngine::CNNNetwork cnnNet,
                                              std::vector<PreProcessInfo>& preProcInfo, bool sharedConstants,
                                              mlir::TimingScope& rootTiming, bool enableProfiling, Logger log) {
    log.setName("IE::FrontEnd::importNetwork");

    validateCNNNetwork(cnnNet);

    log.trace("Load IE::FrontEnd dependent Dialects");
    ctx->loadDialect<IE::IEDialect>();

    const auto netGraph = cnnNet.getFunction();
    VPUX_THROW_UNLESS(netGraph != nullptr, "Old IR versions (prior v10) are not supported : {0}", cnnNet.getName());

    log.trace("Run common nGraph passes");
    runNGraphPasses(netGraph, preProcInfo, rootTiming);

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef(cnnNet.getName()));
    const auto mainFuncName = mlir::FlatSymbolRefAttr::get(ctx, "main");

    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    log.trace("Add CNNNetwork Operation");
    addCNNNetworkOp(builder, mainFuncName, cnnNet, netGraph, rootTiming, enableProfiling);

    log.trace("Import nGraph function");
    NGraphImporter importer(ctx, netGraph, sharedConstants, log);
    importer.buildMainFunc(builder, mainFuncName.getValue(), rootTiming);

    log.trace("Validate MLIR module");
    auto finalTiming = rootTiming.nest("Validate MLIR module");
    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    return module;
}
