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

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux {
namespace VPUIP {

class NCEInvariant final {
public:
    static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

public:
    static mlir::LogicalResult verifyOp(mlir::Operation* op, Logger log = Logger::global());

public:
    static mlir::LogicalResult verifyCMX(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyConvCMX(mlir::Location loc, mlir::ModuleOp module, mlir::ShapedType inputType,
                                             mlir::ShapedType filterType, mlir::ShapedType outputType,
                                             mlir::ArrayAttr kernelStrides, Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module, mlir::ShapedType inputType,
                                             mlir::ShapedType outputType, mlir::ArrayAttr kernelSize,
                                             mlir::ArrayAttr kernelStrides, Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IE::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module,
                                                mlir::ShapedType firstInputType, mlir::ShapedType secondInputType,
                                                mlir::ShapedType outputType, Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyGroupConvCMX(mlir::Location loc, mlir::ModuleOp module, mlir::ShapedType inputType,
                                                  mlir::ShapedType filterType, mlir::ShapedType outputType,
                                                  mlir::ArrayAttr kernelStrides, Logger log = Logger::global());

    static mlir::LogicalResult verifyPrefetchCMX(IE::ConvolutionOp origOp, vpux::OutputTiling tiling,
                                                 Logger log = Logger::global());
    static mlir::LogicalResult verifyPrefetchCMX(IE::MaxPoolOp origOp, vpux::OutputTiling tiling,
                                                 Logger log = Logger::global());
    static mlir::LogicalResult verifyPrefetchCMX(IE::GroupConvolutionOp origOp, vpux::OutputTiling tiling,
                                                 Logger log = Logger::global());

    static mlir::LogicalResult verifyPrefetchCMX(IE::AddOp origOp, vpux::OutputTiling tiling,
                                                 Logger log = Logger::global());
    static mlir::LogicalResult verifyPrefetchCMX(IE::MultiplyOp origOp, vpux::OutputTiling tiling,
                                                 Logger log = Logger::global());
    static mlir::LogicalResult verifyPrefetchCMX(IE::SubtractOp origOp, vpux::OutputTiling tiling,
                                                 Logger log = Logger::global());
    static mlir::LogicalResult verifyPrefetchCMX(IE::AndOp origOp, vpux::OutputTiling tiling,
                                                 Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwisePrefetchCMX(mlir::Operation* op, vpux::OutputTiling tiling,
                                                        Logger log = Logger::global());

    static mlir::LogicalResult verifyPrefetchPatternCMX(mlir::Operation* op, vpux::OutputTiling tiling,
                                                        mlir::Operation* parentOp, vpux::OutputTiling parentTiling,
                                                        Logger log);

public:
    static mlir::LogicalResult verifyChannels(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyConvChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                  mlir::ShapedType filterType, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::AddOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::MultiplyOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::SubtractOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::AndOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyEltwiseChannels(mlir::Location loc, mlir::ShapedType firstInputType,
                                                     mlir::ShapedType secondInputType, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyGroupConvChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                       mlir::ShapedType filterType, Logger log = Logger::global());

public:
    static mlir::LogicalResult verifyKernel(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::ConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::MaxPoolOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::AddOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::MultiplyOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::SubtractOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::AndOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::GroupConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(mlir::Location loc, int64_t KY, int64_t KX, int64_t SY, int64_t SX,
                                            int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                                            VPU::ArchKind arch, Logger log = Logger::global());
};

}  // namespace VPUIP
}  // namespace vpux
