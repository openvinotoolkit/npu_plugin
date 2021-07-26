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

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_VPUIP
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_VPUIP

include "vpux/compiler/dialect/IERT/ops.td"
include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"

//
// Constraints
//

def HasNoValue : Constraint<CPred<"$0 == nullptr">>;

//
// IERT.Copy -> VPUIP.NNDMA
//

def createNNDMAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::NNDMAOp>($_loc, $0, $1)
        }]>;

def createUPADMAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::UPADMAOp>($_loc, $0, $1)
        }]>;

def RewriteCopy :
        Pat<
            (IERT_CopyOp $input, $output),
            (createNNDMAOp $input, $output)
        >;

//
// IERT.Convert -> VPUIP.ConvertUPA
//

def createConvertUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ConvertUPAOp>($_loc, $0, $1)
        }]>;

def RewriteConvert :
        Pat<
            (IERT_ConvertOp $input, $output),
            (createConvertUPAOp $input, $output)
        >;

//
// IERT.SoftMax -> VPUIP.SoftMaxUPA
//

def createSoftMaxUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SoftMaxUPAOp>($_loc, $0, $1, $2)
        }]>;

def RewriteSoftMax :
        Pat<
            (IERT_SoftMaxOp $input, $output, $axisInd),
            (createSoftMaxUPAOp $input, $output, $axisInd)
        >;

//
// IERT.[Max|Avg]Pool -> VPUIP.PoolingUPA
//

class createPoolingUPAOp<string method> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PoolingUPAOp>(
                $_loc, $0, $1,
                vpux::VPUIP::PoolLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::VPUIP::PoolLayerType::}] # method # [{
                ),
                $2, $3, $4, $5,
                nullptr)
        }]>;

def RewriteAvgPool :
        Pat<
            (IERT_AvgPoolOp $input, $output, $kernel_size, $strides, $pads_begin, $pads_end),
            (createPoolingUPAOp<"AVG"> $input, $output, $kernel_size, $strides, $pads_begin, $pads_end)
        >;

def RewriteMaxPool :
        Pat<
            (IERT_MaxPoolOp $input, $output, $kernel_size, $strides, $pads_begin, $pads_end, $post_op),
            (createPoolingUPAOp<"MAX"> $input, $output, $kernel_size, $strides, $pads_begin, $pads_end),
            [
                (HasNoValue $post_op)
            ]
        >;

//
// IERT.Convolution -> VPUIP.ConvolutionUPA
//

def createConvolutionUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ConvolutionUPAOp>(
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4, $5, $6, $7, 1)
        }]>;

def RewriteConvolution :
        Pat<
            (IERT_ConvolutionOp $input, $filter, $bias, $output, $strides, $pads_begin, $pads_end, $dilations, $post_op),
            (createConvolutionUPAOp $input, $filter, $bias, $output, $strides, $dilations, $pads_begin, $pads_end),
            [
                (HasNoValue $post_op)
            ]
        >;

def createGroupConvolutionUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ConvolutionUPAOp>(
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4, $5, $6, $7, $8.getInt())
        }]>;

def HasGroupsAttr : Constraint<CPred<"$0 != nullptr">>;

def RewriteGroupConvolution :
        Pat<
            (IERT_GroupConvolutionOp $input, $filter, $bias, $output, $strides, $pads_begin, $pads_end, $dilations, $groups, $post_op),
            (createGroupConvolutionUPAOp $input, $filter, $bias, $output, $strides, $dilations, $pads_begin, $pads_end, $groups),
            [
                (HasGroupsAttr $groups)
            ]
        >;

//
// IERT.ReLU -> VPUIP.ReLUUPA
//

def createReLUUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ReLUUPAOp>($_loc, $0, $1)
        }]>;

def RewriteReLU :
        Pat<
            (IERT_ReLUOp $input, $output),
            (createReLUUPAOp $input, $output)
        >;

//
// IERT.Sigmoid -> VPUIP.SigmoidUPA
//

def createSigmoidUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SigmoidUPAOp>($_loc, $0, $1)
        }]>;

def RewriteSigmoid :
        Pat<
            (IERT_SigmoidOp $input, $output),
            (createSigmoidUPAOp $input, $output)
        >;

//
// IERT.Clamp -> VPUIP.ClampUPA
//

def createClampUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ClampUPAOp>($_loc, $0, $1, $2, $3)
        }]>;

def RewriteClamp :
        Pat<
            (IERT_ClampOp $input, $output, $min, $max),
            (createClampUPAOp $input, $output, $min, $max)
        >;

//
// IERT.Elu -> VPUIP.EluUPA
//

def createEluUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::EluUPAOp>($_loc, $0, $1, $2)
        }]>;

def RewriteElu :
        Pat<
            (IERT_EluOp $input, $output, $x),
            (createEluUPAOp $input, $output, $x)
        >;

//
// IERT.HSwish -> VPUIP.HSwishUPA
//

def createHSwishUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::HSwishUPAOp>($_loc, $0, $1)
        }]>;

def RewriteHSwish :
        Pat<
            (IERT_HSwishOp $input, $output),
            (createHSwishUPAOp $input, $output)
        >;

//
// IERT.Tanh -> VPUIP.TanhUPA
//

def createTanhUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::TanhUPAOp>($_loc, $0, $1)
        }]>;

def RewriteTanh :
        Pat<
            (IERT_TanhOp $input, $output),
            (createTanhUPAOp $input, $output)
        >;

//
// IERT.Exp -> VPUIP.ExpUPA
//

def createExpUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ExpUPAOp>($_loc, $0, $1)
        }]>;

def RewriteExp :
        Pat<
            (IERT_ExpOp $input, $output),
            (createExpUPAOp $input, $output)
        >;

//
// IERT.PRelu -> VPUIP.PReluUPA
//

def createPReluUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PReluUPAOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewritePRelu :
        Pat<
            (IERT_PReluOp $input, $negative_slope, $output),
            (createPReluUPAOp $input, $negative_slope, $output)
        >;

//
// IERT.LeakyRelu -> VPUIP.LeakyReluUPA
//

def createLeakyReluUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::LeakyReluUPAOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteLeakyRelu :
        Pat<
            (IERT_LeakyReluOp $input, $output, $negative_slope),
            (createLeakyReluUPAOp $input, $output, $negative_slope)
        >;

//
// IERT.Quantize/Dequantize -> VPUIP.QuantCastUPA
//

def createQuantCastUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::QuantCastUPAOp>($_loc, $0, $1)
        }]>;

def RewriteQuantize :
        Pat<
            (IERT_QuantizeOp $input, $output),
            (createQuantCastUPAOp $input, $output)
        >;

def RewriteDequantize :
        Pat<
            (IERT_DequantizeOp $input, $output),
            (createQuantCastUPAOp $input, $output)
        >;

//
// IERT.PerAxisTile -> VPUIP.PerAxisTileUPA
//

def createPerAxisTileUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PerAxisTileUPAOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteTile :
        Pat<
            (IERT_PerAxisTileOp $input, $output, $axis, $tiles),
            (createPerAxisTileUPAOp $input, $output, $axis, $tiles)
        >;

//
// IERT.[Add|Multiply|Divide|SquaredDiff|Power|FloorMod|Min|Max]Pool -> VPUIP.EltwiseUPA
//

class createEltwiseUPAOp<string method> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::EltwiseUPAOp>(
                $_loc, $0, $1, $2,
                vpux::VPUIP::EltwiseLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::VPUIP::EltwiseLayerType::}] # method # [{
                ))
        }]>;

def RewriteAdd :
        Pat<
            (IERT_AddOp $input1, $input2, $output),
            (createEltwiseUPAOp<"ADD"> $input1, $input2, $output)
        >;

def RewriteMultiply :
        Pat<
            (IERT_MultiplyOp $input1, $input2, $output),
            (createEltwiseUPAOp<"MULTIPLY"> $input1, $input2, $output)
        >;

def RewriteDivide :
        Pat<
            (IERT_DivideOp $input1, $input2, $output),
            (createEltwiseUPAOp<"DIVIDE"> $input1, $input2, $output)
        >;

def RewriteSquaredDiff :
        Pat<
            (IERT_SquaredDifferenceOp $input1, $input2, $output),
            (createEltwiseUPAOp<"SQUARED_DIFF"> $input1, $input2, $output)
        >;

def RewritePower :
        Pat<
            (IERT_PowerOp $input1, $input2, $output),
            (createEltwiseUPAOp<"POWER"> $input1, $input2, $output)
        >;

def RewriteFloorMod :
        Pat<
            (IERT_FloorModOp $input1, $input2, $output),
            (createEltwiseUPAOp<"FLOOR_MOD"> $input1, $input2, $output)
        >;

def RewriteMinimum :
        Pat<
            (IERT_MinimumOp $input1, $input2, $output),
            (createEltwiseUPAOp<"MIN"> $input1, $input2, $output)
        >;

def RewriteMaximum :
        Pat<
            (IERT_MaximumOp $input1, $input2, $output),
            (createEltwiseUPAOp<"MAX"> $input1, $input2, $output)
        >;
//
// IERT.GRN -> VPUIP.GRNUPA
//

def createGRNUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::GRNUPAOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteGRN :
        Pat<
            (IERT_GRNOp $input, $output, $bias),
            (createGRNUPAOp $input, $output, $bias)
        >;

//
// IERT.Swish -> VPUIP.SwishUPA
//

def createSwishUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SwishUPAOp>(
                $_loc, $0, $1, $2)
        }]>;

def HasBeta : Constraint<CPred<"$0 != nullptr && $1.empty()">>;

def RewriteSwish :
        Pat<
            (IERT_SwishOp $input, $beta, $output, $beta_value),
            (createSwishUPAOp $input, $output, $beta_value),
            [
                (HasBeta $beta_value, $beta)
            ]>;

//
// IERT.DetectionOutput -> VPUIP.DetectionOutputUPA
//

def createDetectionOutputUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::DetectionOutputUPAOp>(
                $_loc, $0, $1, $2,
                $3.size() == 1 ? $3[0] : nullptr,
                $4.size() == 1 ? $4[0] : nullptr,
                $5, $6)
        }]>;

def RewriteDetectionOutput :
        Pat<
            (IERT_DetectionOutputOp $in_box_logits, $in_class_preds, $in_proposals, $in_additional_preds, $in_additional_proposals, $output, $attr),
            (createDetectionOutputUPAOp $in_box_logits, $in_class_preds, $in_proposals, $in_additional_preds, $in_additional_proposals, $output, $attr)
        >;

//
// IERT.ScaleShift -> VPUIP.ScaleShiftUPA
//

def createScaleShiftUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ScaleShiftUPAOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr,
                $3)
        }]>;

def RewriteScaleShift :
        Pat<
            (IERT_ScaleShiftOp $input, $weights, $biases, $output),
            (createScaleShiftUPAOp $input, $weights, $biases, $output)
        >;

//
// IERT.Negative -> VPUIP.NegativeUPA
//

def createNegativeUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::NegativeUPAOp>(
                $_loc, $0, $1)
        }]>;

def RewriteNegative :
        Pat<
            (IERT_NegativeOp $input, $output),
            (createNegativeUPAOp $input, $output)
        >;

//
// IERT.ROIPooling -> VPUIP.ROIPoolingUPA
//
def createROIPoolingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ROIPoolingUPAOp>(
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;
def RewriteROIPooling :
        Pat<
            (IERT_ROIPoolingOp $input, $coords, $output, $output_size, $spatial_scale, $method),
            (createROIPoolingUPAOp $input, $coords, $output, $output_size, $spatial_scale, $method)
        >;

//
// IERT.Transpose -> VPUIP.Permute
//

def createTransposeUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PermuteUPAOp>(
                $_loc, $0, $1, $2)
        }]>;

def HasConstantOrder : Constraint<CPred<"$0.empty() && $1 != nullptr">>;

def RewriteTranspose :
        Pat<
            (IERT_TransposeOp $input, $order, $output, $order_value),
            (createTransposeUPAOp $input, $output, $order_value),
            [
                (HasConstantOrder $order, $order_value)
            ]
        >;

//
// IERT.Reorder -> VPUIP.Permute
//

def createReorderUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PermuteUPAOp>(
                $_loc, $0, $1, nullptr)
        }]>;

def RewriteReorder :
        Pat<
            (IERT_ReorderOp $input, $output),
            (createReorderUPAOp $input, $output)
        >;

//
// IERT.CTCGreedyDecoder -> VPUIP.CTCGreedyDecoderUPA
//

def createCTCGreedyDecoderUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::CTCGreedyDecoderUPAOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteCTCGreedyDecoder :
        Pat<
            (IERT_CTCGreedyDecoderOp $input, $sequenceLengths, $output, $mergeRepeated),
            (createCTCGreedyDecoderUPAOp $input, $sequenceLengths, $output, $mergeRepeated)
        >;
//
// IERT.Pad -> VPUIP.Pad
//

def createPadUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PadUPAOp>(
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewritePad :
        Pat<
            (IERT_PadOp $input, $output, $pads_begin, $pads_end, $pad_value, $mode),
            (createPadUPAOp $input, $output, $pads_begin, $pads_end, $pad_value, $mode)
        >;
//
// IERT.StridedSlice -> VPUIP.StridedSlice
//

def createStridedSliceUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::StridedSliceUPAOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteStridedSlice :
        Pat<
            (IERT_StridedSliceOp $input, $output, $begins, $ends, $strides),
            (createStridedSliceUPAOp $input, $output, $begins, $ends, $strides)
        >;

//
// IERT.Interpolate -> VPUIP.Interpolate
//

def createInterpolateUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::InterpolateUPAOp>(
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteInterpolate :
        Pat<
            (IERT_InterpolateOp $input, $output, $mode, $coord_mode, $nearest_mode, $antialias),
            (createInterpolateUPAOp $input, $output, $mode, $coord_mode, $nearest_mode, $antialias)
        >;

//
// IERT.RegionYolo -> VPUIP.RegionYoloUPA
//

def createRegionYoloUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::RegionYoloUPAOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6)
        }]>;

def RewriteRegionYolo :
        Pat<
            (IERT_RegionYoloOp $input, $output, $coords, $classes, $regions, $do_softmax, $mask, $axis, $end_axis, $anchors),
            (createRegionYoloUPAOp $input, $output, $coords, $classes, $regions, $do_softmax, $mask)
        >;

#endif
