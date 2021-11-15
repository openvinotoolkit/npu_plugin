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

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_EMU
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_EMU

include "vpux/compiler/dialect/IE/ops.td"
include "vpux/compiler/dialect/EMU/ops.td"

include "mlir/IR/OpBase.td"

//
// Constraints
//

def HasNoValue : Constraint<CPred<"$0 == nullptr">>;

//
// IE.Convert -> EMU.ConvertUPA
//

def createConvertUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ConvertUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteConvert :
        Pat<
            (IE_ConvertOp:$result $input, $dstElemType),
            (createConvertUPAOp $input, $result)
        >;

//
// IE.SoftMax -> EMU.SoftMaxUPA
//

def createSoftMaxUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::SoftMaxUPAOp>($_loc, $2.getType(), $0, $1)
        }]>;

def RewriteSoftMax :
        Pat<
            (IE_SoftMaxOp:$result $input, $axisInd),
            (createSoftMaxUPAOp $input, $axisInd, $result)
        >;

//
// IE.AvgPool -> EMU.PoolingUPA
//

def createAvgPoolingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::PoolingUPAOp>(
                $_loc, $6.getType(), $0,
                vpux::EMU::PoolLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::EMU::PoolLayerType::AVG
                ),
                $1, $2, $3, $4, $5)
        }]>;

def RewriteAvgPool :
        Pat<
            (IE_AvgPoolOp:$result $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $exclude_pads),
            (createAvgPoolingUPAOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $exclude_pads, $result)
        >;

//
// IE.LRN_IE -> EMU.NormUPA
//

def createNormUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::NormUPAOp> (
                $_loc, $6.getType(), $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteLRN_IE :
        Pat<
            (IE_LRN_IEOp:$result $input, $alpha, $beta, $bias, $size, $region),
            (createNormUPAOp $input, $alpha, $beta, $bias, $size, $region, $result)
        >;

//
// IE.Broadcast -> EMU.BroadcastUPA
//

def createBroadcastUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::BroadcastUPAOp> (
                $_loc, $4.getType(), $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3)
        }]>;

def RewriteBroadcast :
        Pat<
            (IE_BroadcastOp:$result $input, $target_shape, $axes_mapping, $mode),
            (createBroadcastUPAOp $input, $target_shape, $axes_mapping, $mode, $result)
        >;

//
// IE.MaxPool -> EMU.PoolingUPA
//

def createMaxPoolingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::PoolingUPAOp>(
                $_loc, $5.getType(), $0,
                vpux::EMU::PoolLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::EMU::PoolLayerType::MAX
                ),
                $1, $2, $3, $4, nullptr)
        }]>;

def RewriteMaxPool :
        Pat<
            (IE_MaxPoolOp:$result $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $post_op),
            (createMaxPoolingUPAOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $result),
            [
                (HasNoValue $post_op)
            ]
        >;

//
// IE.Convolution -> EMU.ConvolutionUPA
//

def createGroupConvolutionUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ConvolutionUPAOp>(
                $_loc, $8.getType(), $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4, $5, $6, $7.getInt())
        }]>;

def HasGroupsAttr : Constraint<CPred<"$0 != nullptr">>;

def RewriteGroupConvolution :
        Pat<
            (IE_GroupConvolutionOp:$result $input, $filter, $bias, $strides, $pads_begin, $pads_end, $dilations, $groups, $post_op),
            (createGroupConvolutionUPAOp $input, $filter, $bias, $strides, $dilations, $pads_begin, $pads_end, $groups, $result),
            [
                (HasGroupsAttr $groups)
            ]
        >;

//
// IE.ReLU -> EMU.ReLUUPA
//

def createReLUUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ReLUUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteReLU :
        Pat<
            (IE_ReLUOp:$result $input),
            (createReLUUPAOp $input, $result)
        >;

//
// IE.Sigmoid -> EMU.SigmoidUPA
//

def createSigmoidUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::SigmoidUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteSigmoid :
        Pat<
            (IE_SigmoidOp:$result $input),
            (createSigmoidUPAOp $input, $result)
        >;

//
// IE.Clamp -> EMU.ClampUPA
//

def createClampUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ClampUPAOp>($_loc, $3.getType(), $0, $1, $2)
        }]>;

def RewriteClamp :
        Pat<
            (IE_ClampOp:$result $input, $min, $max),
            (createClampUPAOp $input, $min, $max, $result)
        >;

//
// IE.Elu -> EMU.EluUPA
//

def createEluUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::EluUPAOp>($_loc, $2.getType(), $0, $1)
        }]>;

def RewriteElu :
        Pat<
            (IE_EluOp:$result $input, $x),
            (createEluUPAOp $input, $x, $result)
        >;

//
// IE.HSwish -> EMU.HSwishUPA
//

def createHSwishUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::HSwishUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteHSwish :
        Pat<
            (IE_HSwishOp:$result $input),
            (createHSwishUPAOp $input, $result)
        >;

//
// IE.Floor -> EMU.FloorUPA
//

def createFloorUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::FloorUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteFloor :
        Pat<
            (IE_FloorOp:$result $input),
            (createFloorUPAOp $input, $result)
        >;

//
// IE.Round -> EMU.RoundUPA
//

def createRoundUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::RoundUPAOp>($_loc, $2.getType(), $0, $1)
        }]>;

def RewriteRound :
        Pat<
            (IE_RoundOp:$result $input, $mode),
            (createRoundUPAOp $input, $mode, $result)
        >;

//
// IE.Mish -> EMU.MishUPA
//

def createMishUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::MishUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteMish :
        Pat<
            (IE_MishOp:$result $input),
            (createMishUPAOp $input, $result)
        >;

//
// IE.Erf -> EMU.ErfUPA
//

def createErfUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ErfUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteErf :
        Pat<
            (IE_ErfOp:$result $input),
            (createErfUPAOp $input, $result)
        >;

//
// IE.Tanh -> EMU.TanhUPA
//

def createTanhUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::TanhUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteTanh :
        Pat<
            (IE_TanhOp:$result $input),
            (createTanhUPAOp $input, $result)
        >;

//
// IE.Sqrt -> EMU.SqrtUPA
//

def createSqrtUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::SqrtUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteSqrt :
        Pat<
            (IE_SqrtOp:$result $input),
            (createSqrtUPAOp $input, $result)
        >;

//
// IE.Log -> EMU.LogUPA
//

def createLogUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::LogUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteLog :
        Pat<
            (IE_LogOp:$result $input),
            (createLogUPAOp $input, $result)
        >;

//
// IE.Exp -> EMU.ExpUPA
//

def createExpUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ExpUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteExp :
        Pat<
            (IE_ExpOp:$result $input),
            (createExpUPAOp $input, $result)
        >;

//
// IE.PRelu -> EMU.PReluUPA
//

def createPReluUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::PReluUPAOp>(
                $_loc, $2.getType(), $0, $1)
        }]>;

def RewritePRelu :
        Pat<
            (IE_PReluOp:$result $input, $negative_slope),
            (createPReluUPAOp $input, $negative_slope, $result)
        >;

//
// IE.LeakyRelu -> EMU.LeakyReluUPA
//

def createLeakyReluUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::LeakyReluUPAOp>(
                $_loc, $2.getType(), $0, $1)
        }]>;

def RewriteLeakyRelu :
        Pat<
            (IE_LeakyReluOp:$result $input, $negative_slope),
            (createLeakyReluUPAOp $input, $negative_slope, $result)
        >;

//
// IE.Quantize/Dequantize -> EMU.QuantCastUPA
//

def createQuantCastUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::QuantCastUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteQuantize :
        Pat<
            (IE_QuantizeOp:$result $input, $dstElemType),
            (createQuantCastUPAOp $input, $result)
        >;

def RewriteDequantize :
        Pat<
            (IE_DequantizeOp:$result $input, $dstElemType),
            (createQuantCastUPAOp $input, $result)
        >;

//
// IE.PerAxisTile -> EMU.PerAxisTileUPA
//

def createPerAxisTileUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::PerAxisTileUPAOp>(
                $_loc, $3.getType(), $0, $1, $2)
        }]>;

def RewriteTile :
        Pat<
            (IE_PerAxisTileOp:$result $input, $axis, $tiles),
            (createPerAxisTileUPAOp $input, $axis, $tiles, $result)
        >;

//
// IE.[Add|Multiply|Divide|SquaredDiff|Power|FloorMod|Min|Max|And|Equal]Pool -> EMU.EltwiseUPA
//

class createEltwiseUPAOp<string method> :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::EltwiseUPAOp>(
                $_loc, $2.getType(), $0, $1,
                vpux::EMU::EltwiseLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::EMU::EltwiseLayerType::}] # method # [{
                ))
        }]>;

def RewriteAdd :
        Pat<
            (IE_AddOp:$result $input1, $input2, $auto_broadcast, $post_op),
            (createEltwiseUPAOp<"ADD"> $input1, $input2, $result),
            [
                (HasNoValue $post_op)
            ]
        >;

def RewriteMultiply :
        Pat<
            (IE_MultiplyOp:$result $input1, $input2, $auto_broadcast, $post_op),
            (createEltwiseUPAOp<"MULTIPLY"> $input1, $input2, $result),
            [
                (HasNoValue $post_op)
            ]
        >;

def RewriteDivide :
        Pat<
            (IE_DivideOp:$result $input1, $input2, $auto_broadcast),
            (createEltwiseUPAOp<"DIVIDE"> $input1, $input2, $result)
        >;

def RewriteSquaredDiff :
        Pat<
            (IE_SquaredDifferenceOp:$result $input1, $input2, $auto_broadcast),
            (createEltwiseUPAOp<"SQUARED_DIFF"> $input1, $input2, $result)
        >;

def RewritePower :
        Pat<
            (IE_PowerOp:$result $input1, $input2, $auto_broadcast),
            (createEltwiseUPAOp<"POWER"> $input1, $input2, $result)
        >;

def RewriteFloorMod :
        Pat<
            (IE_FloorModOp:$result $input1, $input2, $auto_broadcast),
            (createEltwiseUPAOp<"FLOOR_MOD"> $input1, $input2, $result)
        >;

def RewriteMinimum :
        Pat<
            (IE_MinimumOp:$result $input1, $input2, $auto_broadcast),
            (createEltwiseUPAOp<"MIN"> $input1, $input2, $result)
        >;

def RewriteMaximum :
        Pat<
            (IE_MaximumOp:$result $input1, $input2, $auto_broadcast),
            (createEltwiseUPAOp<"MAX"> $input1, $input2, $result)
        >;

def RewriteAnd :
        Pat<
            (IE_AndOp:$result $input1, $input2, $auto_broadcast, $post_op),
            (createEltwiseUPAOp<"AND"> $input1, $input2, $result),
            [
                (HasNoValue $post_op)
            ]
        >;

def RewriteEqual :
        Pat<
            (IE_EqualOp:$result $input1, $input2, $auto_broadcast),
            (createEltwiseUPAOp<"EQUAL"> $input1, $input2, $result)
        >;

//
// IE.GRN -> EMU.GRNUPA
//

def createGRNUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::GRNUPAOp>(
                $_loc, $2.getType(), $0, $1)
        }]>;

def RewriteGRN :
        Pat<
            (IE_GRNOp:$result $input, $bias),
            (createGRNUPAOp $input, $bias, $result)
        >;

//
// IE.Swish -> EMU.SwishUPA
//

def createSwishUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::SwishUPAOp>(
                $_loc, $2.getType(), $0, $1)
        }]>;

def HasBeta : Constraint<CPred<"$0 != nullptr && $1.empty()">>;

def RewriteSwish :
        Pat<
            (IE_SwishOp:$result $input, $beta, $beta_value),
            (createSwishUPAOp $input, $beta_value, $result),
            [
                (HasBeta $beta_value, $beta)
            ]>;

//
// IE.DetectionOutput -> EMU.DetectionOutputUPA
//

def createDetectionOutputUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::DetectionOutputUPAOp>(
                $_loc, $6.getType(), $0, $1, $2,
                $3.size() == 1 ? $3[0] : nullptr,
                $4.size() == 1 ? $4[0] : nullptr,
                $5)
        }]>;

def RewriteDetectionOutput :
        Pat<
            (IE_DetectionOutputOp:$result $in_box_logits, $in_class_preds, $in_proposals, $in_additional_preds, $in_additional_proposals, $attr),
            (createDetectionOutputUPAOp $in_box_logits, $in_class_preds, $in_proposals, $in_additional_preds, $in_additional_proposals, $attr, $result)
        >;

//
// IE.ScaleShift -> EMU.ScaleShiftUPA
//

def createScaleShiftUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ScaleShiftUPAOp>(
                $_loc, $3.getType(), $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr)
        }]>;

def RewriteScaleShift :
        Pat<
            (IE_ScaleShiftOp:$result $input, $weights, $biases),
            (createScaleShiftUPAOp $input, $weights, $biases, $result)
        >;

//
// IE.Negative -> EMU.NegativeUPA
//

def createNegativeUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::NegativeUPAOp>(
                $_loc, $1.getType(), $0)
        }]>;

def RewriteNegative :
        Pat<
            (IE_NegativeOp:$result $input),
            (createNegativeUPAOp $input, $result)
        >;

//
// IE.Proposal -> EMU.ProposalUPA
//
def createProposalUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ProposalUPAOp>(
                $_loc, $4.getType(), $0, $1, $2, $3)
        }]>;
def RewriteProposal :
        Pat<
            (IE_ProposalOp:$result $class_probs, $bbox_deltas, $image_shape, $proposal_attrs),
            (createProposalUPAOp $class_probs, $bbox_deltas, $image_shape, $proposal_attrs, $result)
        >;

//
// IE.ROIPooling -> EMU.ROIPoolingUPA
//
def createROIPoolingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ROIPoolingUPAOp>(
                $_loc, $5.getType(), $0, $1, $2, $3, $4)
        }]>;
def RewriteROIPooling :
        Pat<
            (IE_ROIPoolingOp:$result $input, $coords, $output_size, $spatial_scale, $method),
            (createROIPoolingUPAOp $input, $coords, $output_size, $spatial_scale, $method, $result)
        >;

//
// IE.ROIAlign -> EMU.ROIAlignUPA
//

def createROIAlignUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ROIAlignUPAOp>(
                $_loc, $8.getType(), $0, $1, $2, $3, $4, $5, $6, $7)
        }]>;
def RewriteROIAlign :
        Pat<
            (IE_ROIAlignOp:$result $input, $coords, $roisIdx, $pooled_h, $pooled_w, $sampling_ratio, $spatial_scale, $poolingMode),
            (createROIAlignUPAOp $input, $coords, $roisIdx, $pooled_h, $pooled_w, $sampling_ratio, $spatial_scale, $poolingMode, $result)
        >;

//
// IE.CTCGreedyDecoder -> EMU.CTCGreedyDecoderUPA
//

def createCTCGreedyDecoderUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::CTCGreedyDecoderUPAOp>(
                $_loc, $3.getType(), $0, $1, $2)
        }]>;

def RewriteCTCGreedyDecoder :
        Pat<
            (IE_CTCGreedyDecoderOp:$result $input, $sequenceLengths, $mergeRepeated),
            (createCTCGreedyDecoderUPAOp $input, $sequenceLengths, $mergeRepeated, $result)
        >;
//
// IE.Pad -> EMU.Pad
//

def createPadUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::PadUPAOp>(
                $_loc, $5.getType(), $0, $1, $2, $3, $4)
        }]>;

def RewritePad :
        Pat<
            (IE_PadOp:$result $input, $pads_begin, $pads_end, $pad_value, $pads_begin_attr, $pads_end_attr, $pad_value_attr, $mode),
            (createPadUPAOp $input, $pads_begin_attr, $pads_end_attr, $pad_value_attr, $mode, $result)
        >;
//
// IE.StridedSlice -> EMU.StridedSlice
//

def createStridedSliceUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::StridedSliceUPAOp>(
                $_loc, $4.getType(), $0, $1, $2, $3)
        }]>;

def RewriteStridedSlice :
        Pat<
            (IE_StridedSliceOp:$result $input, $begins, $ends, $strides, $begins_attr, $ends_attr, $strides_attr, $begin_mask, $end_mask, $new_axis_mask, $shrink_axis_mask, $ellipsis_mask),
            (createStridedSliceUPAOp $input, $begins_attr, $ends_attr, $strides_attr, $result)
        >;

//
// IE.Interpolate -> EMU.Interpolate
//

def createInterpolateUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::InterpolateUPAOp>(
                $_loc, $2.getType(), $0, $1.mode().getValue(), $1.coord_mode().getValue(), $1.nearest_mode().getValue(), $1.antialias().getValue())
        }]>;

def RewriteInterpolate :
        Pat<
            (IE_InterpolateOp:$result $input, $sizes, $scales, $axes, $sizes_attr, $scales_attr, $axes_attr, $attr),
            (createInterpolateUPAOp $input, $attr, $result)
        >;

//
// IE.RegionYolo -> EMU.RegionYoloUPA
//

def createRegionYoloUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::RegionYoloUPAOp>(
                $_loc, $6.getType(), $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteRegionYolo :
        Pat<
            (IE_RegionYoloOp:$result $input, $coords, $classes, $regions, $do_softmax, $mask, $axis, $end_axis, $anchors),
            (createRegionYoloUPAOp $input, $coords, $classes, $regions, $do_softmax, $mask, $result)
        >;

//
// IE.MVN -> EMU.MVNUPA
//

def createMVNUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::MVNUPAOp>(
                $_loc, $4.getType(), $0, $1, $2, $3)
        }]>;

def RewriteMVN :
        Pat<
            (IE_MVNOp:$result $input, $across_channels, $normalize_variance, $eps),
            (createMVNUPAOp $input, $across_channels, $normalize_variance, $eps, $result)
        >;

//
// IE.Gather -> EMU.Gather
//

def createGatherUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::GatherUPAOp>(
                $_loc, $3.getType(), $0, $1, $2)
        }]>;

def RewriteGather :
        Pat<
            (IE_GatherOp:$result $input, $indices, $axis, $axis_value),
            (createGatherUPAOp $input, $indices, $axis_value, $result)
        >;

//
// IE.MemPermute -> EMU.Permute
//

def createMemPermuteUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::PermuteUPAOp>(
                $_loc, $2.getType(), $0, $1)
        }]>;

def RewriteMemPermute :
        Pat<
            (IE_MemPermuteOp:$result $input, $dst_order, $mem_perm),
            (createMemPermuteUPAOp $input, $mem_perm, $result)
        >;

//
// IE.Ceiling -> EMU.CeilingUPA
//

def createCeilingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::CeilingUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteCeiling :
        Pat<
            (IE_CeilingOp:$result $input),
            (createCeilingUPAOp $input, $result)
        >;

//
// IE.NormalizeL2 -> EMU.NormalizeIEUPA
//

def createNormalizeIEUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::NormalizeIEUPAOp>(
                $_loc, $5.getType(), $0, $1, $2, $3, $4)
        }]>;

def RewriteNormalizeIE :
        Pat<
            (IE_NormalizeIEOp:$result $data, $weights, $eps, $across_spatial, $channel_shared),
            (createNormalizeIEUPAOp $data, $weights, $eps, $across_spatial, $channel_shared, $result)
        >;

//
// IE.Expand -> EMU.Pad
//

def createExpandAsPadUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::PadUPAOp>(
                $_loc, $3.getType(), $0, $1, $2, getFPAttr($_builder.getContext(), 0),
                vpux::IE::PadModeAttr::get(
                    $_builder.getContext(),
                    vpux::IE::PadMode::CONSTANT
                ))
        }]>;

def RewriteExpand :
        Pat<
            (IE_ExpandOp:$result $input, $pads_begin, $pads_end),
            (createExpandAsPadUPAOp $input, $pads_begin, $pads_end, $result)
        >;

//
// IE.Slice -> EMU.Slice
//

def createSliceUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::SliceUPAOp>($_loc, $3.getType(), $0, $1, $2)
        }]>;

def RewriteSlice :
        Pat<
            (IE_SliceOp:$result $input, $static_offsets, $static_sizes),
            (createSliceUPAOp $input, $static_offsets, $static_sizes, $result)
        >;

//
// IE.Concat -> EMU.Concat
//

def createConcatUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ConcatUPAOp>($_loc, $3.getType(), $0, $1, $2)
        }]>;

def RewriteConcat :
        Pat<
            (IE_ConcatOp:$result $input, $per_axis, $static_offsets),
            (createConcatUPAOp $input, $per_axis, $static_offsets, $result)
        >;

//
// IE.Reshape -> EMU.Reshape
//

def createReshapeUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::ReshapeUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteReshape :
        Pat<
            (IE_ReshapeOp:$result $input, $shape, $special_zero, $shape_value),
            (createReshapeUPAOp $input, $result)
        >;

//
// IE.QuantCast -> EMU.Copy
//

def createCopyUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::CopyUPAOp>($_loc, $1.getType(), $0)
        }]>;

def RewriteQuantizeCast :
        Pat<
            (IE_QuantizeCastOp:$result $input, $dstElemType),
            (createCopyUPAOp $input, $result)
        >;

//
// IE::PermuteCast -> EMU.Copy
//

def RewritePermuteCast :
        Pat<
            (IE_PermuteCastOp:$result $input, $dst_order, $mem_perm),
            (createMemPermuteUPAOp $input, $mem_perm, $result)
        >;

//
// IE.Squeeze -> EMU.Reshape
//

def RewriteSqueeze :
        Pat<
            (IE_SqueezeOp:$result $input, $axes, $axes_value),
            (createReshapeUPAOp $input, $result)
        >;

//
// IE.Unsqueeze -> EMU.Reshape
//

def RewriteUnsqueeze :
        Pat<
            (IE_UnsqueezeOp:$result $input, $axes, $axes_value),
            (createReshapeUPAOp $input, $result)
        >;

#endif
