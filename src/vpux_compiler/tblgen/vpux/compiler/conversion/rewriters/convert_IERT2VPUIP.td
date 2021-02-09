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

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_IERT2VPUIP
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_IERT2VPUIP

include "vpux/compiler/dialect/IERT/ops.td"
include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"

//
// linalg.copy -> VPUIP.NNDMA
//

def NullAttr : Attr<CPred<"$_self == nullptr">>;

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
            (CopyOp $input, $output, NullAttr:$inputPermutation, NullAttr:$outputPermutation),
            (createNNDMAOp $input, $output)
        >;

//
// IERT.StaticAlloc -> VPUIP.DeclareTensor
//

def createDeclareTensorOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::DeclareTensorOp>(
                $_loc, $0.getType(),
                vpux::VPUIP::MemoryLocation::VPU_DDR_Heap,
                $1.getInt())
        }]>;

def HasMemSpaceDDR :
        Constraint<CPred<[{
            $0.getType().cast<mlir::MemRefType>().getMemorySpace() ==
                    vpux::VPUIP::PhysicalMemoryAttr::get($0.getContext(), vpux::VPUIP::PhysicalMemory::DDR)
        }]>>;

def RewriteStaticAlloc :
        Pat<
            (IERT_StaticAllocOp:$memref $offset),
            (createDeclareTensorOp $memref, $offset),
            [
                (HasMemSpaceDDR $memref)
            ]
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
            (IERT_MaxPoolOp $input, $output, $kernel_size, $strides, $pads_begin, $pads_end),
            (createPoolingUPAOp<"MAX"> $input, $output, $kernel_size, $strides, $pads_begin, $pads_end)
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
            (IERT_ConvolutionOp $input, $filter, $bias, $output, $strides, $pads_begin, $pads_end, $dilations),
            (createConvolutionUPAOp $input, $filter, $bias, $output, $strides, $dilations, $pads_begin, $pads_end)
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
            (IERT_GroupConvolutionOp $input, $filter, $bias, $output, $strides, $pads_begin, $pads_end, $dilations, $groups),
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

#endif
