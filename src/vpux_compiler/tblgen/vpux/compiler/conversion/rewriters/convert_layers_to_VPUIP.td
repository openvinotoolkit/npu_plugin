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
// IERT.AdaAvgPool -> VPUIP.PoolingUPA
//

def RewriteAdaAvgPool :
        Pat<
            (IERT_AdaAvgPoolOp $input1, $input2, $output),
            (createAvgPoolingUPAOp<"ADA_AVG_POOL"> $input1, $input2, $output)
        >;

//
// IERT.AvgPool -> VPUIP.PoolingUPA
//

def createAvgPoolingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PoolingUPAOp>(
                $_loc, $0, $1,
                vpux::VPUIP::PoolLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::VPUIP::PoolLayerType::AVG
                ),
                $2, $3, $4, $5, $6)
        }]>;

def RewriteAvgPool :
        Pat<
            (IERT_AvgPoolOp $input, $output, $kernel_size, $strides, $pads_begin, $pads_end, $exclude_pads),
            (createAvgPoolingUPAOp $input, $output, $kernel_size, $strides, $pads_begin, $pads_end, $exclude_pads)
        >;

//
// IERT.LRN_IE -> VPUIP.NormUPA
//

def createNormUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::NormUPAOp> (
                $_loc, $0, $1, $2, $3, $4, $5, $6)
        }]>;

def RewriteLRN_IE :
        Pat<
            (IERT_LRN_IEOp $input, $output, $alpha, $beta, $bias, $size, $region),
            (createNormUPAOp $input, $output, $alpha, $beta, $bias, $size, $region)
        >;

//
// IERT.Broadcast -> VPUIP.BroadcastUPA
//

def createBroadcastUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::BroadcastUPAOp> (
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4)
        }]>;

def RewriteBroadcast :
        Pat<
            (IERT_BroadcastOp $input, $target_shape, $axes_mapping, $output, $mode),
            (createBroadcastUPAOp $input, $target_shape, $axes_mapping, $output, $mode)
        >;

//
// IERT.Reduce[Max|Mean|Sum]Op -> VPUIP.ReduceUPA
//

class createReduceUPAOp<string method> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ReduceUPAOp>(
                $_loc, $0, $1, $2, $3,
                vpux::VPUIP::ReduceLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::VPUIP::ReduceLayerType::}] # method # [{
                ))
        }]>;

def RewriteReduceMax :
        Pat<
            (IERT_ReduceMaxOp $input, $axes, $output, $keep_dims),
            (createReduceUPAOp<"MAX"> $input, $axes, $output, $keep_dims)
        >;

def RewriteReduceMean :
        Pat<
            (IERT_ReduceMeanOp $input, $axes, $output, $keep_dims),
            (createReduceUPAOp<"MEAN"> $input, $axes, $output, $keep_dims)
        >;

def RewriteReduceSum :
        Pat<
            (IERT_ReduceSumOp $input, $axes, $output, $keep_dims),
            (createReduceUPAOp<"SUM"> $input, $axes, $output, $keep_dims)
        >;

//
// IERT.MaxPool -> VPUIP.PoolingUPA
//

def createMaxPoolingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PoolingUPAOp>(
                $_loc, $0, $1,
                vpux::VPUIP::PoolLayerTypeAttr::get(
                    $_builder.getContext(),
                    vpux::VPUIP::PoolLayerType::MAX
                ),
                $2, $3, $4, $5, nullptr)
        }]>;

def RewriteMaxPool :
        Pat<
            (IERT_MaxPoolOp $input, $output, $kernel_size, $strides, $pads_begin, $pads_end, $post_op),
            (createMaxPoolingUPAOp $input, $output, $kernel_size, $strides, $pads_begin, $pads_end),
            [
                (HasNoValue $post_op)
            ]
        >;

//
// IERT.Convolution -> VPUIP.ConvolutionUPA
//

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
// IERT.Floor -> VPUIP.FloorUPA
//

def createFloorUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::FloorUPAOp>($_loc, $0, $1)
        }]>;

def RewriteFloor :
        Pat<
            (IERT_FloorOp $input, $output),
            (createFloorUPAOp $input, $output)
        >;

//
// IERT.Round -> VPUIP.RoundUPA
//

def createRoundUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::RoundUPAOp>($_loc, $0, $1, $2)
        }]>;

def RewriteRound :
        Pat<
            (IERT_RoundOp $input, $output, $mode),
            (createRoundUPAOp $input, $output, $mode)
        >;

//
// IERT.Mish -> VPUIP.MishUPA
//

def createMishUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::MishUPAOp>($_loc, $0, $1)
        }]>;

def RewriteMish :
        Pat<
            (IERT_MishOp $input, $output),
            (createMishUPAOp $input, $output)
        >;

//
// IERT.Erf -> VPUIP.ErfUPA
//

def createErfUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ErfUPAOp>($_loc, $0, $1)
        }]>;

def RewriteErf :
        Pat<
            (IERT_ErfOp $input, $output),
            (createErfUPAOp $input, $output)
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
// IERT.Sqrt -> VPUIP.SqrtUPA
//

def createSqrtUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SqrtUPAOp>($_loc, $0, $1)
        }]>;

def RewriteSqrt :
        Pat<
            (IERT_SqrtOp $input, $output),
            (createSqrtUPAOp $input, $output)
        >;

//
// IERT.Sinh -> VPUIP.SinhUPA
//

def createSinhUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SinhUPAOp>($_loc, $0, $1)
        }]>;

def RewriteSinh :
        Pat<
            (IERT_SinhOp $input, $output),
            (createSinhUPAOp $input, $output)
        >;

//
// IERT.Cosh -> VPUIP.CoshUPA
//

def createCoshUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::CoshUPAOp>($_loc, $0, $1)
        }]>;

def RewriteCosh :
        Pat<
            (IERT_CoshOp $input, $output),
            (createCoshUPAOp $input, $output)
        >;

//
// IERT.Asinh -> VPUIP.AsinhUPA
//

def createAsinhUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::AsinhUPAOp>($_loc, $0, $1)
        }]>;

def RewriteAsinh :
        Pat<
            (IERT_AsinhOp $input, $output),
            (createAsinhUPAOp $input, $output)
        >;


//
// IERT.Acosh -> VPUIP.AcoshUPA
//

def createAcoshUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::AcoshUPAOp>($_loc, $0, $1)
        }]>;

def RewriteAcosh :
        Pat<
            (IERT_AcoshOp $input, $output),
            (createAcoshUPAOp $input, $output)
        >;


//
// IERT.Log -> VPUIP.LogUPA
//

def createLogUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::LogUPAOp>($_loc, $0, $1)
        }]>;

def RewriteLog :
        Pat<
            (IERT_LogOp $input, $output),
            (createLogUPAOp $input, $output)
        >;

//
// IERT.Gelu -> VPUIP.GeluUPA
//

def createGeluUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::GeluUPAOp>($_loc, $0, $1)
        }]>;

def RewriteGelu :
        Pat<
            (IERT_GeluOp $input, $output),
            (createGeluUPAOp $input, $output)
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
// IERT.[Add|Multiply|Divide|SquaredDiff|Power|FloorMod|Min|Max|And|Equal|Less|LessEqual|NotEqual|Greater|GreaterEqual|LogicalOr|LogicalXor|AdaAvgPool]Pool -> VPUIP.EltwiseUPA
//

class createEltwiseUPAOp<string method> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::EltwiseUPAOp>(
                $_loc, $0, $1, $2,
                vpux::VPU::EltwiseTypeAttr::get(
                    $_builder.getContext(),
                    vpux::VPU::EltwiseType::}] # method # [{
                ))
        }]>;

def RewriteAdd :
        Pat<
            (IERT_AddOp $input1, $input2, $output, $post_op),
            (createEltwiseUPAOp<"ADD"> $input1, $input2, $output),
            [
                (HasNoValue $post_op)
            ]
        >;

def RewriteMultiply :
        Pat<
            (IERT_MultiplyOp $input1, $input2, $output, $post_op),
            (createEltwiseUPAOp<"MULTIPLY"> $input1, $input2, $output),
            [
                (HasNoValue $post_op)
            ]
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

def RewriteAnd :
        Pat<
            (IERT_AndOp $input1, $input2, $output, $post_op),
            (createEltwiseUPAOp<"AND"> $input1, $input2, $output),
            [
                (HasNoValue $post_op)
            ]
        >;

def RewriteEqual :
        Pat<
            (IERT_EqualOp $input1, $input2, $output),
            (createEltwiseUPAOp<"EQUAL"> $input1, $input2, $output)
        >;

def RewriteLess :
        Pat<
            (IERT_LessOp $input1, $input2, $output),
            (createEltwiseUPAOp<"LESS"> $input1, $input2, $output)
        >;

def RewriteLessEqual :
        Pat<
            (IERT_LessEqualOp $input1, $input2, $output),
            (createEltwiseUPAOp<"LESS_EQUAL"> $input1, $input2, $output)
        >;
def RewriteLogicalOr :
        Pat<
            (IERT_LogicalOrOp $input1, $input2, $output),
            (createEltwiseUPAOp<"LOGICAL_OR"> $input1, $input2, $output)
        >;

def RewriteLogicalXor :
        Pat<
            (IERT_LogicalXorOp $input1, $input2, $output),
            (createEltwiseUPAOp<"LOGICAL_XOR"> $input1, $input2, $output)
        >;

def RewriteNotEqual :
        Pat<
            (IERT_NotEqualOp $input1, $input2, $output),
            (createEltwiseUPAOp<"NOT_EQUAL"> $input1, $input2, $output)
        >;

def RewriteGreater :
        Pat<
            (IERT_GreaterOp $input1, $input2, $output),
            (createEltwiseUPAOp<"GREATER"> $input1, $input2, $output)
        >;

def RewriteGreaterEqual :
        Pat<
            (IERT_GreaterEqualOp $input1, $input2, $output),
            (createEltwiseUPAOp<"GREATER_EQUAL"> $input1, $input2, $output)
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
// IERT.Upsampling -> VPUIP.UpsamplingUPA
//

def createUpsamplingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::UpsamplingUPAOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteUpsampling :
        Pat<
            (IERT_UpsamplingOp $input, $upsampling_factor, $pad_l, $pad_r, $output),
            (createUpsamplingUPAOp $input, $upsampling_factor, $pad_l, $pad_r, $output)
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
// IERT.Sign -> VPUIP.SignUPA
//

def createSignUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SignUPAOp>(
                $_loc, $0, $1)
        }]>;

def RewriteSign :
        Pat<
            (IERT_SignOp $input, $output),
            (createSignUPAOp $input, $output)
        >;

//
// IERT.Proposal -> VPUIP.ProposalUPA
//
def createProposalUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ProposalUPAOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;
def RewriteProposal :
        Pat<
            (IERT_ProposalOp $class_probs, $bbox_deltas, $image_shape, $output, $proposal_attrs),
            (createProposalUPAOp $class_probs, $bbox_deltas, $image_shape, $output, $proposal_attrs)
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
// IERT.ROIAlign -> VPUIP.ROIAlignUPA
//

def createROIAlignUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ROIAlignUPAOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7, $8)
        }]>;
def RewriteROIAlign :
        Pat<
            (IERT_ROIAlignOp $input, $coords, $roisIdx, $output, $pooled_h, $pooled_w, $sampling_ratio, $spatial_scale, $poolingMode),
            (createROIAlignUPAOp $input, $coords, $roisIdx, $output, $pooled_h, $pooled_w, $sampling_ratio, $spatial_scale, $poolingMode)
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

//
// IERT.MVN -> VPUIP.MVNUPA
//

def createMVNUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::MVNUPAOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteMVN :
        Pat<
            (IERT_MVNOp $input, $output, $across_channels, $normalize_variance, $eps),
            (createMVNUPAOp $input, $output, $across_channels, $normalize_variance, $eps)
        >;

//
// IERT.Gather -> VPUIP.Gather
//

def createGatherUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::GatherUPAOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteGather :
        Pat<
            (IERT_GatherOp $input, $indices, $axis, $output, $axis_value, $batch_dims),
            (createGatherUPAOp $input, $indices, $output, $axis_value, $batch_dims)
        >;

//
// IERT.YuvToRgb -> VPUIP.YuvToRgbUPA
//

def createYuvToRgbUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::YuvToRgbUPAOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4, $5)
        }]>;

def RewriteYuvToRgb :
        Pat<
            (IERT_YuvToRgbOp $input1, $input2, $input3, $output, $inFmt, $outFmt),
            (createYuvToRgbUPAOp $input1, $input2, $input3, $output, $inFmt, $outFmt)
        >;

//
// IERT.GatherElements -> VPUIP.GatherElements
//

def createGatherElementsUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::GatherElementsUPAOp>(
                    $_loc, $0, $1, $2, $3)
        }]>;

def RewriteGatherElements :
        Pat<
            (IERT_GatherElementsOp $input, $indices, $output, $axis),
            (createGatherElementsUPAOp $input, $indices, $output, $axis)
        >;

//
// IERT.SpaceToDepth -> VPUIP.SpaceToDepth
//

def createSpaceToDepthUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SpaceToDepthUPAOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteSpaceToDepth :
        Pat<
            (IERT_SpaceToDepthOp $input, $output, $block_size, $mode),
            (createSpaceToDepthUPAOp $input, $output, $block_size, $mode)
        >;

//
// IERT.ScatterNDUpdate -> VPUIP.ScatterNDUpdate
//

def createScatterNDUpdateUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ScatterNDUpdateUPAOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteScatterNDUpdate :
        Pat<
            (IERT_ScatterNDUpdateOp $input, $indices, $updates, $output),
            (createScatterNDUpdateUPAOp $input, $indices, $updates, $output)
        >;

//
// IERT.MemPermute -> VPUIP.Permute
//

def createMemPermuteUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::PermuteUPAOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteMemPermute :
        Pat<
            (IERT_MemPermuteOp $input, $output, $mem_perm),
            (createMemPermuteUPAOp $input, $output, $mem_perm)
            >;

//
// IERT.SoftPlus -> VPUIP.SoftPlusUPA
//

def createSoftPlusUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::SoftPlusUPAOp>($_loc, $0, $1)
        }]>;

def RewriteSoftPlus :
        Pat<
            (IERT_SoftPlusOp $input, $output),
            (createSoftPlusUPAOp $input, $output)
        >;

//
// IERT.Ceiling -> VPUIP.CeilingUPA
//

def createCeilingUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::CeilingUPAOp>($_loc, $0, $1)
        }]>;

def RewriteCeiling :
        Pat<
            (IERT_CeilingOp $input, $output),
            (createCeilingUPAOp $input, $output)
        >;

//
// IERT.NormalizeL2 -> VPUIP.NormalizeIEUPA
//

def createNormalizeIEUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::NormalizeIEUPAOp>(
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteNormalizeIE :
        Pat<
            (IERT_NormalizeIEOp $data, $weights, $output, $eps, $across_spatial, $channel_shared),
            (createNormalizeIEUPAOp $data, $weights, $output, $eps, $across_spatial, $channel_shared)
        >;

//
// IERT.ReverseSequence -> VPUIP.ReverseSequenceUPA
//

def createReverseSequenceUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::ReverseSequenceUPAOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteReverseSequence :
        Pat<
            (IERT_ReverseSequenceOp $data, $seq_length, $output, $seq_axis, $batch_axis),
            (createReverseSequenceUPAOp $data, $seq_length, $output, $seq_axis, $batch_axis)
        >;

//
// IERT.DepthToSpace -> VPUIP.DepthToSpaceUPA
//

def createDepthToSpaceUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::DepthToSpaceUPAOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteDepthToSpace :
        Pat<
            (IERT_DepthToSpaceOp $input, $output, $block_size, $mode),
            (createDepthToSpaceUPAOp $input, $output, $block_size, $mode)
        >;

#endif
